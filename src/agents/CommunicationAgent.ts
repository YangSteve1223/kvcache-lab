/**
 * Communication Cost Agent - 通信成本评估Agent
 * 
 * 职责：评估每个KV的访问通信成本
 * 输出：CommunicationState（写入Global State Store）
 * 
 * 核心能力：
 * - 计算token级别的访问成本
 * - 评估层级别的平均成本
 * - 监控带宽利用率
 * - 检测拥塞级别
 * - 预估传输延迟
 * 
 * 这是Transmission-Aware Attention的基础：
 * 让attention第一次具备systems awareness
 */

// ============================================
// 类型定义
// ============================================

/**
 * Token存储位置枚举
 */
export type TokenLocation = 'gpu_hbm' | 'cpu_ram' | 'remote_gpu' | 'compressed';

/**
 * 拥塞级别枚举
 */
export type CongestionLevel = 'low' | 'medium' | 'high';

/**
 * Communication Agent的输入接口
 * 从Global State Store读取相关状态
 */
export interface CommunicationAgentInput {
  /** token索引 -> 存储位置 */
  tokenLocations: Map<number, TokenLocation>;
  /** 带宽(bytes/ms) */
  bandwidthBytesPerMs: number;
  /** 正在传输的KV数量 */
  pendingTransfers: number;
  /** GPU显存总量(bytes) */
  gpuMemoryBytes: number;
  /** 当前GPU显存使用量(bytes) */
  currentMemoryUsage: number;
  /** 模型层数 */
  numLayers: number;
  /** 每层token数量 */
  tokensPerLayer: number;
  /** 每个token的KV大小(bytes) */
  kvBytesPerToken: number;
}

/**
 * Communication Agent的输出状态
 * 写入Global State Store供其他Agent使用
 */
export interface CommunicationState {
  /** token索引 -> 访问成本(ms) */
  tokenAccessCosts: Map<number, number>;
  /** 层索引 -> 平均访问成本(ms) */
  layerAccessCosts: Map<number, number>;
  /** 带宽利用率 0-1 */
  bandwidthUtilization: number;
  /** 拥塞级别 */
  congestionLevel: CongestionLevel;
  /** 预估传输延迟(ms) */
  estimatedTransferLatency: number;
}

// ============================================
// 常量定义
// ============================================

/**
 * 不同存储位置的访问成本基准(ms)
 * 基于典型硬件性能：
 * - GPU HBM: ~1μs，本地显存
 * - CPU RAM: ~50μs，PCIe传输
 * - Remote GPU: ~0.5ms，RDMA/网络传输
 * - Compressed: ~1ms，解压+传输
 */
const ACCESS_COSTS: Record<TokenLocation, number> = {
  gpu_hbm: 0.001,      // ~1μs
  cpu_ram: 0.05,       // ~50μs
  remote_gpu: 0.5,     // ~0.5ms
  compressed: 1.0,     // ~1ms
};

/**
 * 拥塞系数 - 拥塞时访问延迟会放大
 */
const CONGESTION_MULTIPLIERS: Record<CongestionLevel, number> = {
  low: 1.0,
  medium: 1.5,
  high: 3.0,
};

/**
 * β系数 - Transmission-Aware Attention中的成本衰减系数
 * 根据拥塞级别自适应调整
 */
const BETA_COEFFICIENTS: Record<CongestionLevel, number> = {
  low: 0.5,
  medium: 1.0,
  high: 2.0,
};

// ============================================
// Communication Agent实现
// ============================================

export class CommunicationAgent {
  /**
   * 评估通信成本 - 主入口
   * 
   * @param input - 输入参数
   * @returns CommunicationState - 通信状态评估结果
   */
  assess(input: CommunicationAgentInput): CommunicationState {
    const { 
      tokenLocations, 
      bandwidthBytesPerMs, 
      pendingTransfers, 
      kvBytesPerToken 
    } = input;

    // 1. 计算每个token的访问成本
    const tokenAccessCosts = this.computeTokenAccessCosts(
      tokenLocations,
      bandwidthBytesPerMs,
      kvBytesPerToken,
      pendingTransfers
    );

    // 2. 计算每层的平均访问成本
    const layerAccessCosts = this.computeLayerAccessCosts(
      tokenAccessCosts,
      input.numLayers,
      input.tokensPerLayer
    );

    // 3. 评估带宽利用率和拥塞级别
    const { utilization, congestionLevel } = this.assessBandwidth(
      pendingTransfers,
      bandwidthBytesPerMs,
      kvBytesPerToken
    );

    // 4. 预估传输延迟
    const estimatedTransferLatency = this.estimateTransferLatency(
      kvBytesPerToken,
      bandwidthBytesPerMs,
      congestionLevel
    );

    return {
      tokenAccessCosts,
      layerAccessCosts,
      bandwidthUtilization: utilization,
      congestionLevel,
      estimatedTransferLatency,
    };
  }

  /**
   * 计算所有token的访问成本
   */
  private computeTokenAccessCosts(
    tokenLocations: Map<number, TokenLocation>,
    bandwidthBytesPerMs: number,
    kvBytesPerToken: number,
    pendingTransfers: number
  ): Map<number, number> {
    const costs = new Map<number, number>();
    
    // 根据pending transfers计算拥塞系数
    const congestionLevel = this.determineCongestionLevel(pendingTransfers, bandwidthBytesPerMs);
    const congestionMultiplier = CONGESTION_MULTIPLIERS[congestionLevel];

    for (const [tokenIndex, location] of tokenLocations) {
      costs.set(
        tokenIndex,
        this.computeTokenAccessCost(
          tokenIndex,
          location,
          bandwidthBytesPerMs,
          kvBytesPerToken,
          congestionMultiplier
        )
      );
    }

    return costs;
  }

  /**
   * 计算单个token的访问成本
   * 
   * 公式: cost = base_cost + (kv_size / bandwidth) × congestion_multiplier
   * 
   * @param tokenIndex - token索引
   * @param location - 存储位置
   * @param bandwidthBytesPerMs - 带宽
   * @param kvBytesPerToken - KV大小
   * @param congestionMultiplier - 拥塞系数
   */
  private computeTokenAccessCost(
    tokenIndex: number,
    location: TokenLocation,
    bandwidthBytesPerMs: number,
    kvBytesPerToken: number,
    congestionMultiplier: number
  ): number {
    // 获取基础访问成本
    const baseCost = ACCESS_COSTS[location];

    // 计算传输成本（如果需要传输）
    let transferCost = 0;
    if (location === 'remote_gpu' || location === 'compressed') {
      // 传输时间 = 数据大小 / 带宽
      transferCost = (kvBytesPerToken / bandwidthBytesPerMs) * congestionMultiplier;
    }

    // 如果是压缩格式，还需要考虑解压成本
    let decompressCost = 0;
    if (location === 'compressed') {
      // 解压成本约为压缩数据的0.5ms
      decompressCost = 0.5 * congestionMultiplier;
    }

    return baseCost + transferCost + decompressCost;
  }

  /**
   * 计算每层的平均访问成本
   * 
   * @param tokenAccessCosts - token级别的访问成本
   * @param numLayers - 层数
   * @param tokensPerLayer - 每层token数量
   */
  private computeLayerAccessCosts(
    tokenAccessCosts: Map<number, number>,
    numLayers: number,
    tokensPerLayer: number
  ): Map<number, number> {
    const layerCosts = new Map<number, number>();

    for (let layer = 0; layer < numLayers; layer++) {
      const layerStart = layer * tokensPerLayer;
      const layerEnd = layerStart + tokensPerLayer;
      
      let sumCost = 0;
      let count = 0;

      for (let tokenIndex = layerStart; tokenIndex < layerEnd; tokenIndex++) {
        const cost = tokenAccessCosts.get(tokenIndex);
        if (cost !== undefined) {
          sumCost += cost;
          count++;
        }
      }

      // 层平均成本 = 该层所有token成本之和 / token数量
      layerCosts.set(layer, count > 0 ? sumCost / count : 0);
    }

    return layerCosts;
  }

  /**
   * 评估带宽利用率和拥塞级别
   * 
   * @param pendingTransfers - 待传输数量
   * @param bandwidthBytesPerMs - 带宽
   * @param kvBytesPerToken - 平均传输大小
   */
  private assessBandwidth(
    pendingTransfers: number,
    bandwidthBytesPerMs: number,
    kvBytesPerToken: number
  ): { utilization: number; congestionLevel: CongestionLevel } {
    if (pendingTransfers === 0) {
      return { utilization: 0, congestionLevel: 'low' };
    }

    // 计算总传输需求
    const totalTransferDemand = pendingTransfers * kvBytesPerToken;
    
    // 计算带宽利用时间窗口（假设1秒内的带宽）
    const bandwidthWindowBytes = bandwidthBytesPerMs * 1000;
    
    // 利用率 = 需求 / 可用带宽
    const utilization = Math.min(1, totalTransferDemand / bandwidthWindowBytes);

    // 拥塞级别判断
    let congestionLevel: CongestionLevel;
    if (utilization < 0.3) {
      congestionLevel = 'low';
    } else if (utilization < 0.7) {
      congestionLevel = 'medium';
    } else {
      congestionLevel = 'high';
    }

    return { utilization, congestionLevel };
  }

  /**
   * 预估传输延迟
   * 
   * @param transferSizeBytes - 传输大小
   * @param bandwidthBytesPerMs - 带宽
   * @param congestionLevel - 拥塞级别
   */
  private estimateTransferLatency(
    transferSizeBytes: number,
    bandwidthBytesPerMs: number,
    congestionLevel: CongestionLevel
  ): number {
    // 基础传输延迟
    const baseLatency = transferSizeBytes / bandwidthBytesPerMs;
    
    // 应用拥塞系数
    const congestionMultiplier = CONGESTION_MULTIPLIERS[congestionLevel];

    // 加上固定开销（约0.1ms）
    return baseLatency * congestionMultiplier + 0.1;
  }

  /**
   * 根据pending transfers和带宽判断拥塞级别
   */
  private determineCongestionLevel(
    pendingTransfers: number,
    bandwidthBytesPerMs: number
  ): CongestionLevel {
    if (pendingTransfers === 0) return 'low';
    
    // 假设高带宽(>100GB/s)可以处理更多并发
    const highBandwidthThreshold = 100 * 1024 * 1024; // 100MB/ms = 100GB/s
    
    if (bandwidthBytesPerMs > highBandwidthThreshold) {
      // 高带宽场景
      if (pendingTransfers < 10) return 'low';
      if (pendingTransfers < 50) return 'medium';
      return 'high';
    } else {
      // 普通带宽场景
      if (pendingTransfers < 5) return 'low';
      if (pendingTransfers < 20) return 'medium';
      return 'high';
    }
  }
}

// ============================================
// Transmission-Aware Attention工具函数
// ============================================

/**
 * 计算Transmission-Aware Attention Scores
 * 
 * 核心公式: modified_score[i] = relevance[i] × exp(-β × cost[i])
 * 
 * β根据拥塞级别自适应：
 * - low congestion: β=0.5 (几乎只看relevance)
 * - medium: β=1.0 (适度考虑cost)
 * - high: β=2.0 (强烈考虑cost)
 * 
 * @param relevance - 原始注意力分数
 * @param costs - 访问成本数组
 * @param congestionLevel - 拥塞级别
 */
export function computeTransmissionAwareScores(
  relevance: Float64Array | number[],
  costs: Float64Array | number[],
  congestionLevel: CongestionLevel
): number[] {
  const beta = BETA_COEFFICIENTS[congestionLevel];
  
  // 归一化成本到[0,1]范围
  const maxCost = Math.max(...costs);
  const normalizedCosts = costs.map(c => c / maxCost);
  
  // 应用成本衰减
  const modifiedScores = relevance.map((r, i) => 
    r * Math.exp(-beta * normalizedCosts[i])
  );
  
  // 重新归一化确保和为1
  const sum = modifiedScores.reduce((a, b) => a + b, 0);
  return modifiedScores.map(s => sum > 0 ? s / sum : 0);
}

/**
 * 计算拥塞感知的查询延迟
 * 
 * @param baseLatencyMs - 基础延迟
 * @param congestionLevel - 拥塞级别
 * @param queueLength - 当前队列长度
 */
export function computeCongestionAwareLatency(
  baseLatencyMs: number,
  congestionLevel: CongestionLevel,
  queueLength: number
): number {
  const multiplier = CONGESTION_MULTIPLIERS[congestionLevel];
  // 队列延迟 = 基础延迟 × 拥塞系数 × (1 + 队列长度/10)
  return baseLatencyMs * multiplier * (1 + queueLength / 10);
}

/**
 * 获取β系数
 */
export function getBetaCoefficient(congestionLevel: CongestionLevel): number {
  return BETA_COEFFICIENTS[congestionLevel];
}

// ============================================
// 导出默认实例
// ============================================

export const communicationAgent = new CommunicationAgent();
