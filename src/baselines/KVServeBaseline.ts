/**
 * KVServe Baseline - 简化版实现
 * 
 * KVServe核心思想（基于论文）：
 * 1. 策略空间搜索：在离线阶段，遍历多种压缩配置，构建Pareto集（压缩率×延迟×质量）
 * 2. 贝叶斯优化：用少量采样近似Pareto前沿
 * 3. 在线控制器：根据当前带宽/SLO/负载选择Pareto集中的最优配置
 * 
 * 简化实现：
 * - 均匀压缩策略（不区分P/D端）
 * - 离线搜索Pareto集，在线快速选择
 * - 根据带宽和SLO约束选择最优配置
 */

// 类型定义
export interface CompressionParams {
  totalLayers: number;
  totalTokens: number;
  bandwidthBytesPerMs: number;
  gpuMemoryBytes: number;
  currentMemoryUsage: number;
  taskType: string;
  sloLatencyMs?: number;
  prefixHitRate?: number;
}

export interface CompressionOutput {
  strategy: string;
  totalLayers: number;
  pLayerRetention: number[];
  dLayerRetention: number[];
  pKeyPrecision: number[];
  pValuePrecision: number[];
  dKeyPrecision: number[];
  dValuePrecision: number[];
  avgCompressionRatio: number;
  estimatedBandwidthSaving: number;
}

export interface ICompressionStrategy {
  readonly name: string;
  readonly type: string;
  computeConfig(params: CompressionParams): CompressionOutput;
  estimateQualityImpact(config: CompressionOutput, taskType: string): number;
}

// 工具函数
function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

// Pareto点接口
interface ParetoPoint {
  retention: number;
  compressionRatio: number;
  quality: number;
  latencyMs: number;
}

/**
 * KVServeBaseline - KVServe简化版
 * 
 * 核心特点：
 * - Uniform压缩：所有层使用相同保留率
 * - 离线Pareto搜索：搜索不同保留率配置的性能权衡
 * - 在线快速选择：根据当前约束选择最优Pareto点
 */
export class KVServeBaseline implements ICompressionStrategy {
  readonly name = 'KVServeBaseline';
  readonly type = 'baseline';
  
  // 预计算的Pareto集
  private paretoSet: ParetoPoint[] = [];
  private isParetoSetBuilt: boolean = false;
  
  // 精度常量
  private readonly KEY_PRECISION = 8;   // FP8
  private readonly VALUE_PRECISION = 4; // INT4
  
  /**
   * 离线阶段：构建Pareto集
   * 
   * 搜索策略：
   * - 保留率范围：0.3 - 1.0，步长0.1
   * - 对每种配置估算：compression_ratio, quality, latency
   * - 筛选非支配解（Pareto前沿）
   */
  buildParetoSet(params: CompressionParams): void {
    const { totalTokens, bandwidthBytesPerMs, gpuMemoryBytes, currentMemoryUsage } = params;
    
    // 计算KV大小估算
    const kvBytesPerToken = 1024; // 1KB per token
    const originalSizeBytes = totalTokens * kvBytesPerToken * totalTokens; // 简化估算
    
    // 生成候选配置
    const candidates: ParetoPoint[] = [];
    
    for (let retention = 0.3; retention <= 1.0; retention += 0.1) {
      retention = Math.round(retention * 100) / 100;
      
      // 计算压缩后大小
      const precisionRatio = (this.KEY_PRECISION / 16) * (this.VALUE_PRECISION / 16);
      const compressedRatio = retention * precisionRatio;
      
      // 计算传输延迟
      const transferBytes = originalSizeBytes * compressedRatio;
      const transferTimeMs = transferBytes / bandwidthBytesPerMs;
      
      // 计算质量（保留率越高质量越好）
      const quality = retention;
      
      // 计算总延迟（prefill时间 + 传输时间）
      const prefillBaseMs = 50;
      const prefillPerTokenMs = 0.01;
      const prefillTimeMs = prefillBaseMs + totalTokens * prefillPerTokenMs;
      const latencyMs = prefillTimeMs + transferTimeMs;
      
      // 计算显存压力调整
      const memoryUsageRatio = currentMemoryUsage / gpuMemoryBytes;
      const memoryPenalty = memoryUsageRatio > 0.8 ? 1.2 : 1.0;
      
      candidates.push({
        retention,
        compressionRatio: round4(compressedRatio),
        quality: round4(quality),
        latencyMs: round4(latencyMs * memoryPenalty)
      });
    }
    
    // 筛选Pareto前沿（非支配解）
    this.paretoSet = this.filterParetoFront(candidates);
    this.isParetoSetBuilt = true;
  }
  
  /**
   * 筛选Pareto前沿
   * 
   * 支配定义：
   * - 点A支配点B：A在所有维度都不差于B，且至少在一个维度严格更好
   * - Pareto前沿：不被任何其他点支配的点的集合
   * 
   * 优化目标：最大化compressionRatio（越小越好），最大化quality，最小化latency
   */
  private filterParetoFront(candidates: ParetoPoint[]): ParetoPoint[] {
    const paretoFront: ParetoPoint[] = [];
    
    for (const candidate of candidates) {
      let isDominated = false;
      
      for (const other of candidates) {
        if (candidate === other) continue;
        
        // 检查是否被支配
        // A支配B：compression更好(更小) 或 quality更好 或 latency更低，且其他不更差
        const betterCompression = candidate.compressionRatio <= other.compressionRatio;
        const betterQuality = candidate.quality >= other.quality;
        const betterLatency = candidate.latencyMs <= other.latencyMs;
        const atLeastOneStrictlyBetter = 
          candidate.compressionRatio < other.compressionRatio ||
          candidate.quality > other.quality ||
          candidate.latencyMs < other.latencyMs;
        
        if (betterCompression && betterQuality && betterLatency && atLeastOneStrictlyBetter) {
          isDominated = true;
          break;
        }
      }
      
      if (!isDominated) {
        paretoFront.push(candidate);
      }
    }
    
    // 按compressionRatio排序
    return paretoFront.sort((a, b) => a.compressionRatio - b.compressionRatio);
  }
  
  /**
   * 在线阶段：从Pareto集中选择最优配置
   * 
   * 选择标准：
   * 1. 满足SLO约束（延迟 < sloLatencyMs）
   * 2. 在满足SLO的前提下，最大化质量
   * 3. 如果没有满足SLO的配置，选择延迟最接近SLO的配置
   */
  computeConfig(params: CompressionParams): CompressionOutput {
    const { totalLayers, sloLatencyMs = 2000, bandwidthBytesPerMs } = params;
    
    // 确保Pareto集已构建
    if (!this.isParetoSetBuilt) {
      this.buildParetoSet(params);
    }
    
    // 从Pareto集中选择最优配置
    // 策略：找到满足SLO且质量最高的配置
    let selectedPoint: ParetoPoint | null = null;
    let minLatencyDiff = Infinity;
    
    for (const point of this.paretoSet) {
      if (point.latencyMs <= sloLatencyMs) {
        // 满足SLO，选择质量最高的
        if (selectedPoint === null || point.quality > selectedPoint.quality) {
          selectedPoint = point;
        }
      }
    }
    
    // 如果没有满足SLO的配置，选择延迟最接近SLO的配置
    if (selectedPoint === null) {
      for (const point of this.paretoSet) {
        const latencyDiff = Math.abs(point.latencyMs - sloLatencyMs);
        if (latencyDiff < minLatencyDiff) {
          minLatencyDiff = latencyDiff;
          selectedPoint = point;
        }
      }
    }
    
    // 默认配置（如果Pareto集为空）
    if (selectedPoint === null) {
      selectedPoint = {
        retention: 0.5,
        compressionRatio: 0.125,
        quality: 0.5,
        latencyMs: 500
      };
    }
    
    // 构建返回配置
    const retention = selectedPoint.retention;
    const precisionRatio = (this.KEY_PRECISION / 16) * (this.VALUE_PRECISION / 16);
    
    // 初始化数组
    const pLayerRetention: number[] = [];
    const dLayerRetention: number[] = [];
    const pKeyPrecision: number[] = [];
    const pValuePrecision: number[] = [];
    const dKeyPrecision: number[] = [];
    const dValuePrecision: number[] = [];
    
    // 所有层使用相同配置（Uniform压缩）
    for (let i = 0; i < totalLayers; i++) {
      pLayerRetention.push(retention);
      dLayerRetention.push(Math.max(retention, 0.6)); // D端不低于0.6
      pKeyPrecision.push(this.KEY_PRECISION);
      pValuePrecision.push(this.VALUE_PRECISION);
      dKeyPrecision.push(16); // D端使用FP16
      dValuePrecision.push(8); // D端使用FP8
    }
    
    return {
      strategy: this.name,
      totalLayers,
      pLayerRetention,
      dLayerRetention,
      pKeyPrecision,
      pValuePrecision,
      dKeyPrecision,
      dValuePrecision,
      avgCompressionRatio: selectedPoint.compressionRatio,
      estimatedBandwidthSaving: round4(1 - selectedPoint.compressionRatio)
    };
  }
  
  /**
   * 预估质量影响
   */
  estimateQualityImpact(config: CompressionOutput, taskType: string): number {
    const avgRetention = config.pLayerRetention.reduce((a, b) => a + b, 0) / config.totalLayers;
    return round4(avgRetention);
  }
  
  /**
   * 获取Pareto集（用于调试和分析）
   */
  getParetoSet(): ParetoPoint[] {
    return [...this.paretoSet];
  }
}
