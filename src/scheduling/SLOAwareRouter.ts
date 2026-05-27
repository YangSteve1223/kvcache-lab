/**
 * SLO感知路由调度器
 * 
 * 核心功能：
 * 1. 根据SLO约束选择最优压缩策略
 * 2. 动态调整压缩激进程度
 * 3. 多目标权衡（延迟vs质量vs吞吐量）
 * 
 * 设计思想：
 * - SLO越紧 → 压缩越激进（优先满足延迟）
 * - SLO越松 → 保守压缩（优先保证质量）
 * - 负载越高 → 适当增加压缩激进程度
 */

import { ICompressionStrategy } from './ICompressionStrategy.js';

// ========== 类型定义 ==========

/**
 * SLO配置接口
 */
export interface SLOConfig {
  sloTTFTMs: number;        // TTFT SLO目标 (ms)
  sloE2EMs: number;         // E2E SLO目标 (ms)
  bandwidthBytesPerMs: number; // 当前带宽 (bytes/ms)
  currentLoad: number;      // 当前负载 0-1
}

/**
 * 策略性能预估接口
 */
export interface StrategyEstimate {
  strategy: ICompressionStrategy;
  estimatedTTFT: number;
  estimatedE2E: number;
  estimatedQuality: number;
  compressionRatio: number;
  meetsSLO: boolean;
}

/**
 * 路由决策接口
 */
export interface RoutingDecision {
  selectedStrategy: ICompressionStrategy;
  estimatedTTFT: number;
  estimatedE2E: number;
  estimatedQuality: number;
  compressionRatio: number;
  aggressivenessLevel: 'low' | 'medium' | 'high';
  reasoning: string;
}

// ========== SLO感知路由器 ==========

/**
 * SLOAwareRouter - SLO感知路由调度器
 * 
 * 根据SLO约束和当前系统状态，动态选择最优压缩策略
 */
export class SLOAwareRouter {
  private strategies: ICompressionStrategy[] = [];
  private defaultStrategy: ICompressionStrategy | null = null;
  
  /**
   * 注册可用的压缩策略
   */
  registerStrategy(strategy: ICompressionStrategy, isDefault: boolean = false): void {
    this.strategies.push(strategy);
    if (isDefault || this.strategies.length === 1) {
      this.defaultStrategy = strategy;
    }
  }
  
  /**
   * 根据SLO约束选择最优策略
   * 
   * 选择逻辑：
   * 1. 估算各策略的TTFT和E2E
   * 2. 筛选满足SLO约束的策略
   * 3. 在满足SLO的策略中，选择质量最高的
   * 4. 如果没有策略满足SLO，选择延迟最接近SLO的策略
   * 
   * @param config SLO配置
   * @param params 压缩参数（用于计算各策略配置）
   * @returns 路由决策
   */
  selectStrategy(config: SLOConfig, params: CompressionParams): RoutingDecision {
    const { sloTTFTMs, sloE2EMs, currentLoad } = config;
    
    // 估算各策略性能
    const estimates: StrategyEstimate[] = this.strategies.map(strategy => {
      return this.estimateStrategyPerformance(strategy, config, params);
    });
    
    // 筛选满足SLO的策略
    const meetingSLOStrategies = estimates.filter(e => e.meetsSLO);
    
    let selectedEstimate: StrategyEstimate;
    let reasoning: string;
    
    if (meetingSLOStrategies.length > 0) {
      // 满足SLO的策略存在：选择质量最高的
      meetingSLOStrategies.sort((a, b) => b.estimatedQuality - a.estimatedQuality);
      selectedEstimate = meetingSLOStrategies[0];
      reasoning = `策略 ${selectedEstimate.strategy.name} 满足SLO且质量最高`;
    } else {
      // 没有策略满足SLO：选择TTFT最接近SLO的
      estimates.sort((a, b) => 
        Math.abs(a.estimatedTTFT - sloTTFTMs) - Math.abs(b.estimatedTTFT - sloTTFTMs)
      );
      selectedEstimate = estimates[0];
      reasoning = `无策略满足SLO，选择延迟最接近的 ${selectedEstimate.strategy.name}`;
    }
    
    // 计算激进程度
    const aggressivenessLevel = this.calculateAggressiveness(config, selectedEstimate);
    
    return {
      selectedStrategy: selectedEstimate.strategy,
      estimatedTTFT: selectedEstimate.estimatedTTFT,
      estimatedE2E: selectedEstimate.estimatedE2E,
      estimatedQuality: selectedEstimate.estimatedQuality,
      compressionRatio: selectedEstimate.compressionRatio,
      aggressivenessLevel,
      reasoning
    };
  }
  
  /**
   * 估算单个策略的性能
   */
  private estimateStrategyPerformance(
    strategy: ICompressionStrategy,
    config: SLOConfig,
    params: CompressionParams
  ): StrategyEstimate {
    const { sloTTFTMs, sloE2EMs, bandwidthBytesPerMs } = config;
    
    // 计算压缩配置
    const compressionOutput = strategy.computeConfig(params);
    
    // 估算TTFT
    // TTFT = prefill_time + transfer_time
    const prefillBaseMs = 50;
    const prefillPerTokenMs = 0.01;
    const prefillTimeMs = prefillBaseMs + params.totalTokens * prefillPerTokenMs;
    
    // 计算KV大小和传输时间
    const kvBytesPerToken = 1024;
    const originalSizeBytes = params.totalTokens * kvBytesPerToken * params.totalLayers;
    const compressedRatio = compressionOutput.avgCompressionRatio;
    const transferTimeMs = (originalSizeBytes * compressedRatio) / bandwidthBytesPerMs;
    
    const estimatedTTFT = prefillTimeMs + transferTimeMs;
    
    // 估算E2E（简化为 TTFT + output_tokens * decode_time）
    const decodeTimePerTokenMs = 0.5; // TPOT约500ms/token
    const estimatedE2E = estimatedTTFT + params.outputTokens * decodeTimePerTokenMs;
    
    // 估算质量
    const estimatedQuality = strategy.estimateQualityImpact(compressionOutput, params.taskType);
    
    // 判断是否满足SLO
    const meetsSLO = estimatedTTFT <= sloTTFTMs && estimatedE2E <= sloE2EMs;
    
    return {
      strategy,
      estimatedTTFT: Math.round(estimatedTTFT * 100) / 100,
      estimatedE2E: Math.round(estimatedE2E * 100) / 100,
      estimatedQuality: Math.round(estimatedQuality * 10000) / 10000,
      compressionRatio: compressedRatio,
      meetsSLO
    };
  }
  
  /**
   * 计算压缩激进程度
   */
  private calculateAggressiveness(
    config: SLOConfig,
    estimate: StrategyEstimate
  ): 'low' | 'medium' | 'high' {
    const { sloTTFTMs, currentLoad } = config;
    
    // 基于SLO紧迫度和负载计算激进程度
    // SLO越紧，激进程度越高
    const sloUrgency = 1 - (estimate.estimatedTTFT / sloTTFTMs);
    
    // 负载越高，激进程度越高
    const loadFactor = currentLoad;
    
    // 综合评分
    const score = sloUrgency * 0.6 + loadFactor * 0.4;
    
    if (score > 0.7) return 'high';
    if (score > 0.4) return 'medium';
    return 'low';
  }
  
  /**
   * 动态调整压缩激进程度
   * 
   * 根据当前负载动态调整压缩配置
   * - load > 0.8 → 激进压缩
   * - load < 0.5 → 保守压缩
   * - 0.5-0.8 → 线性插值
   * 
   * @param load 当前负载 (0-1)
   * @param baseConfig 基础压缩配置
   * @returns 调整后的压缩配置
   */
  adjustAggressiveness(
    load: number,
    baseConfig: CompressionParams
  ): number {
    // 限制在[0,1]范围
    load = Math.max(0, Math.min(1, load));
    
    // 激进程度因子
    let aggressiveness: number;
    
    if (load > 0.8) {
      // 高负载：激进压缩
      aggressiveness = 0.8 + (load - 0.8) * 1.0;
    } else if (load < 0.5) {
      // 低负载：保守压缩
      aggressiveness = 0.3 + load * 0.4;
    } else {
      // 中等负载：线性插值
      aggressiveness = 0.5 + (load - 0.5) * 0.6;
    }
    
    return Math.round(aggressiveness * 100) / 100;
  }
  
  /**
   * 获取所有策略的预估性能
   */
  getAllEstimates(config: SLOConfig, params: CompressionParams): StrategyEstimate[] {
    return this.strategies.map(strategy => {
      return this.estimateStrategyPerformance(strategy, config, params);
    });
  }
  
  /**
   * 获取默认策略
   */
  getDefaultStrategy(): ICompressionStrategy | null {
    return this.defaultStrategy;
  }
  
  /**
   * 获取已注册策略列表
   */
  getRegisteredStrategies(): string[] {
    return this.strategies.map(s => s.name);
  }
}

// ========== 压缩参数类型 ==========

export interface CompressionParams {
  totalLayers: number;
  totalTokens: number;
  bandwidthBytesPerMs: number;
  gpuMemoryBytes: number;
  currentMemoryUsage: number;
  taskType: string;
  sloLatencyMs?: number;
  outputTokens?: number;
  prefixHitRate?: number;
}

// ========== 工具函数 ==========

/**
 * 简化版SLO检查
 * 判断给定配置是否满足SLO约束
 */
export function checkSLOCompliance(
  ttft: number,
  e2e: number,
  sloTTFT: number,
  sloE2E: number
): { compliant: boolean; ttftDiff: number; e2eDiff: number } {
  const ttftDiff = ttft - sloTTFT;
  const e2eDiff = e2e - sloE2E;
  
  return {
    compliant: ttft <= sloTTFT && e2e <= sloE2E,
    ttftDiff: Math.round(ttftDiff * 100) / 100,
    e2eDiff: Math.round(e2eDiff * 100) / 100
  };
}

/**
 * 计算SLO满足率
 */
export function calculateSLOSatisfactionRate(
  results: Array<{ ttftMs: number; e2eLatencyMs: number }>,
  sloTTFT: number,
  sloE2E: number
): number {
  const compliantCount = results.filter(r => 
    r.ttftMs <= sloTTFT && r.e2eLatencyMs <= sloE2E
  ).length;
  
  return Math.round((compliantCount / results.length) * 10000) / 10000;
}
