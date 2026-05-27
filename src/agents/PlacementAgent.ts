/**
 * KV Placement Agent - KV放置管理Agent
 * 
 * 职责：管理KV在存储层级之间的放置和迁移
 * 输出：PlacementState（写入Global State Store）
 * 
 * 核心能力：
 * - 决定KV的存储层级（GPU/CPU/Remote/Compressed）
 * - 管理KV迁移队列
 * - 监控各层级内存利用率
 * - 类似操作系统的Virtual Memory Manager
 */

import type { TokenLocation } from './CommunicationAgent.js';

// ============================================
// 类型定义
// ============================================

/**
 * Reuse预测结果
 */
export interface ReusePrediction {
  /** 预估重用距离 */
  reuseDistance: number;
  /** 重用概率 0-1 */
  reuseProbability: number;
}

/**
 * 内存利用率信息
 */
export interface MemoryUtilizationInfo {
  gpuHBM: number;    // bytes used
  cpuRAM: number;    // bytes used
  remote: number;    // bytes used
}

/**
 * 内存容量信息
 */
export interface MemoryCapacityInfo {
  gpuHBM: number;    // bytes total
  cpuRAM: number;    // bytes total
  remote: number;    // bytes total
}

/**
 * Placement Agent的输入接口
 * 从Global State Store读取相关状态
 */
export interface PlacementAgentInput {
  /** token索引 -> 当前存储位置 */
  currentLocations: Map<number, TokenLocation>;
  /** token索引 -> Reuse预测 */
  reusePredictions: Map<number, ReusePrediction>;
  /** token索引 -> 访问成本 */
  accessCosts: Map<number, number>;
  /** 各层内存使用量(bytes) */
  memoryUtilization: MemoryUtilizationInfo;
  /** 各层内存容量(bytes) */
  memoryCapacity: MemoryCapacityInfo;
}

/**
 * 迁移计划项
 */
export interface MigrationItem {
  tokenId: number;
  from: TokenLocation;
  to: TokenLocation;
  priority: number;  // 优先级，数值越大优先级越高
}

/**
 * 层级内存利用率
 */
export interface LayerMemoryUtilization {
  gpuHBM: number;    // 0-1
  cpuRAM: number;    // 0-1
  remote: number;    // 0-1
}

/**
 * Placement Agent的输出状态
 * 写入Global State Store供其他Agent使用
 */
export interface PlacementState {
  /** token索引 -> 目标存储位置 */
  tokenLocations: Map<number, TokenLocation>;
  /** 各层内存利用率 0-1 */
  memoryUtilization: LayerMemoryUtilization;
  /** 迁移队列 */
  migrationQueue: MigrationItem[];
}

// ============================================
// 常量定义
// ============================================

/**
 * 内存压力阈值
 */
const MEMORY_PRESSURE_THRESHOLDS = {
  GPU_HIGH: 0.8,   // GPU内存>80%时开始降级
  GPU_LOW: 0.5,    // GPU内存<50%时开始升级
  CPU_HIGH: 0.85,  // CPU内存>85%时开始降级
  CPU_LOW: 0.6,   // CPU内存<60%时开始升级
};

/**
 * 重用距离阈值
 */
const REUSE_DISTANCE_THRESHOLDS = {
  HOT: 3,      // <=3层，热token
  WARM: 10,    // <=10层，温token
  COLD: 50,    // <=50层，冷token
};

/**
 * 存储层级优先级（数值越高越优先）
 */
const STORAGE_PRIORITY: Record<TokenLocation, number> = {
  gpu_hbm: 4,
  cpu_ram: 3,
  remote_gpu: 2,
  compressed: 1,
};

/**
 * 每个token的KV大小默认值(bytes)
 */
const DEFAULT_KV_BYTES_PER_TOKEN = 1024;

// ============================================
// Placement Agent实现
// ============================================

export class PlacementAgent {
  private kvBytesPerToken: number;

  constructor(kvBytesPerToken: number = DEFAULT_KV_BYTES_PER_TOKEN) {
    this.kvBytesPerToken = kvBytesPerToken;
  }

  /**
   * 规划KV放置 - 主入口
   * 
   * @param input - 输入参数
   * @returns PlacementState - 放置状态和迁移计划
   */
  plan(input: PlacementAgentInput): PlacementState {
    const { 
      currentLocations, 
      reusePredictions, 
      accessCosts,
      memoryUtilization: utilization,
      memoryCapacity 
    } = input;

    // 1. 计算目标放置位置
    const targetLocations = this.computeTargetPlacements(
      currentLocations,
      reusePredictions,
      accessCosts,
      utilization,
      memoryCapacity
    );

    // 2. 根据内存压力调整放置
    const adjustedLocations = this.adjustForMemoryPressure(
      targetLocations,
      utilization
    );

    // 3. 生成迁移计划
    const migrationQueue = this.planMigrations(
      currentLocations,
      adjustedLocations
    );

    // 4. 计算新的内存利用率
    const newUtilization = this.computeNewUtilization(
      adjustedLocations,
      memoryCapacity
    );

    return {
      tokenLocations: adjustedLocations,
      memoryUtilization: newUtilization,
      migrationQueue,
    };
  }

  /**
   * 计算目标放置位置
   */
  private computeTargetPlacements(
    currentLocations: Map<number, TokenLocation>,
    reusePredictions: Map<number, ReusePrediction>,
    accessCosts: Map<number, number>,
    utilization: MemoryUtilizationInfo,
    memoryCapacity: MemoryCapacityInfo
  ): Map<number, TokenLocation> {
    const placements = new Map<number, TokenLocation>();

    // 计算当前GPU内存压力
    const gpuPressure = utilization.gpuHBM / memoryCapacity.gpuHBM;
    const cpuPressure = utilization.cpuRAM / memoryCapacity.cpuRAM;

    for (const [tokenId, location] of currentLocations) {
      const reusePrediction = reusePredictions.get(tokenId);
      const accessCost = accessCosts.get(tokenId);

      placements.set(
        tokenId,
        this.computePlacement(
          tokenId,
          reusePrediction || { reuseDistance: 100, reuseProbability: 0 },
          accessCost || 1.0,
          { gpuHBM: gpuPressure, cpuRAM: cpuPressure }
        )
      );
    }

    return placements;
  }

  /**
   * 计算单个token的放置位置
   * 
   * 决策逻辑：
   * - 热token(reuseDistance<=3) + 高reuse概率 → GPU HBM
   * - 温token(reuseDistance<=10) + 中reuse概率 → CPU RAM
   * - 冷token(reuseDistance<=50) + 低reuse概率 → Remote/Compressed
   * 
   * @param tokenId - token索引
   * @param reusePrediction - 重用预测
   * @param accessCost - 访问成本
   * @param memoryPressure - 各层内存压力
   */
  private computePlacement(
    tokenId: number,
    reusePrediction: ReusePrediction,
    accessCost: number,
    memoryPressure: { gpuHBM: number; cpuRAM: number }
  ): TokenLocation {
    const { reuseDistance, reuseProbability } = reusePrediction;
    const { gpuHBM: gpuPressure } = memoryPressure;

    // 计算各层级得分
    // GPU得分：高reuse概率加分，高内存压力减分
    const gpuScore = reuseProbability * 2 - (gpuPressure > 0.8 ? 0.5 : 0);
    
    // CPU得分：中等reuse概率加分
    const cpuScore = reuseProbability * 1.2;
    
    // Remote得分：较低，需要高reuse距离才考虑
    const remoteScore = reuseDistance <= 50 ? 0.3 : 0.1;
    
    // Compressed得分：最低，用于极冷数据
    const compressedScore = 0.1;

    // 基于重用距离的决策
    if (reuseDistance <= REUSE_DISTANCE_THRESHOLDS.HOT && gpuScore > 1.0) {
      return 'gpu_hbm';
    }
    
    if (reuseDistance <= REUSE_DISTANCE_THRESHOLDS.WARM && cpuScore > 0.5) {
      return 'cpu_ram';
    }
    
    if (reuseDistance <= REUSE_DISTANCE_THRESHOLDS.COLD) {
      return 'remote_gpu';
    }
    
    // 最冷的数据使用压缩存储
    return 'compressed';
  }

  /**
   * 生成迁移计划
   * 
   * @param currentLocations - 当前放置
   * @param targetLocations - 目标放置
   */
  private planMigrations(
    currentLocations: Map<number, TokenLocation>,
    targetLocations: Map<number, TokenLocation>
  ): MigrationItem[] {
    const migrations: MigrationItem[] = [];

    for (const [tokenId, targetLocation] of targetLocations) {
      const currentLocation = currentLocations.get(tokenId);
      
      // 只处理需要迁移的token
      if (currentLocation !== targetLocation) {
        // 计算迁移优先级
        const priority = this.calculateMigrationPriority(
          tokenId,
          currentLocation!,
          targetLocation,
          targetLocations
        );

        migrations.push({
          tokenId,
          from: currentLocation!,
          to: targetLocation,
          priority,
        });
      }
    }

    // 按优先级降序排序
    migrations.sort((a, b) => b.priority - a.priority);

    return migrations;
  }

  /**
   * 计算迁移优先级
   * 
   * 优先级规则：
   * - 高优先级迁移：GPU→其他（释放GPU内存）
   * - 中优先级迁移：其他→GPU（提升访问速度）
   * - 低优先级迁移：同类存储间迁移
   */
  private calculateMigrationPriority(
    tokenId: number,
    from: TokenLocation,
    to: TokenLocation,
    allTargets: Map<number, TokenLocation>
  ): number {
    let priority = 0;

    // 基础优先级差值
    priority += STORAGE_PRIORITY[to] - STORAGE_PRIORITY[from];

    // GPU降级（释放GPU内存）优先级更高
    if (from === 'gpu_hbm' && to !== 'gpu_hbm') {
      priority += 3;
    }

    // 升级到GPU优先级次之
    if (from !== 'gpu_hbm' && to === 'gpu_hbm') {
      priority += 2;
    }

    // 考虑该token的访问频率
    // 访问频率高的token升级优先级更高
    // (这里简化处理，实际可以从Global State读取访问频率)

    return Math.max(0, priority);
  }

  /**
   * 根据内存压力调整放置
   * 
   * 内存压力感知规则：
   * - GPU内存>80% → 开始降级（GPU→CPU）
   * - GPU内存<50% → 开始升级（CPU→GPU）
   */
  private adjustForMemoryPressure(
    placements: Map<number, TokenLocation>,
    utilization: MemoryUtilizationInfo
  ): Map<number, TokenLocation> {
    const adjusted = new Map(placements);
    
    const gpuUtilization = utilization.gpuHBM;
    const gpuCapacity = utilization.gpuHBM; // 实际需要传入memoryCapacity
    const cpuUtilization = utilization.cpuRAM;
    const cpuCapacity = utilization.cpuRAM;

    // 简化的内存压力计算（实际需要从memoryCapacity计算）
    const gpuPressure = gpuUtilization; // 这里用绝对值，实际应该用比例
    const cpuPressure = cpuUtilization;

    // GPU内存过高，需要降级一些token
    if (gpuPressure > MEMORY_PRESSURE_THRESHOLDS.GPU_HIGH) {
      this.performDemotion(adjusted, 'gpu_hbm', 'cpu_ram');
    }

    // GPU内存过低，可以升级一些token
    if (gpuPressure < MEMORY_PRESSURE_THRESHOLDS.GPU_LOW && cpuPressure > MEMORY_PRESSURE_THRESHOLDS.CPU_LOW) {
      this.performPromotion(adjusted, 'cpu_ram', 'gpu_hbm');
    }

    return adjusted;
  }

  /**
   * 执行降级操作
   */
  private performDemotion(
    placements: Map<number, TokenLocation>,
    from: TokenLocation,
    to: TokenLocation
  ): void {
    for (const [tokenId, location] of placements) {
      if (location === from) {
        placements.set(tokenId, to);
        break; // 每次只降级一个
      }
    }
  }

  /**
   * 执行升级操作
   */
  private performPromotion(
    placements: Map<number, TokenLocation>,
    from: TokenLocation,
    to: TokenLocation
  ): void {
    for (const [tokenId, location] of placements) {
      if (location === from) {
        placements.set(tokenId, to);
        break; // 每次只升级一个
      }
    }
  }

  /**
   * 计算新的内存利用率
   */
  private computeNewUtilization(
    locations: Map<number, TokenLocation>,
    memoryCapacity: MemoryCapacityInfo
  ): LayerMemoryUtilization {
    let gpuCount = 0;
    let cpuCount = 0;
    let remoteCount = 0;

    for (const location of locations.values()) {
      switch (location) {
        case 'gpu_hbm':
          gpuCount++;
          break;
        case 'cpu_ram':
          cpuCount++;
          break;
        case 'remote_gpu':
        case 'compressed':
          remoteCount++;
          break;
      }
    }

    return {
      gpuHBM: Math.min(1, (gpuCount * this.kvBytesPerToken) / memoryCapacity.gpuHBM),
      cpuRAM: Math.min(1, (cpuCount * this.kvBytesPerToken) / memoryCapacity.cpuRAM),
      remote: Math.min(1, (remoteCount * this.kvBytesPerToken) / memoryCapacity.remote),
    };
  }

  /**
   * 获取当前KV大小
   */
  getKVBytesPerToken(): number {
    return this.kvBytesPerToken;
  }

  /**
   * 设置KV大小
   */
  setKVBytesPerToken(bytes: number): void {
    this.kvBytesPerToken = bytes;
  }
}

// ============================================
// 导出默认实例
// ============================================

export const placementAgent = new PlacementAgent();
