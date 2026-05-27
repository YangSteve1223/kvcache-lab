// 增强版PD分离模拟器
// 专门为KV压缩场景设计，支持压缩对传输时间、质量的影响

import {
  ServingRequest,
  ServingResult,
  SimulatorConfig,
  SimulationStats,
  CompressionConfig,
  TaskType,
  TaskType as TT
} from './types.ts';
import { MathUtils, DeterministicRandom } from './utils.ts';
import { 
  computeQualityScore, 
  computeAverageRetention 
} from './QualityModel.ts';
import { KVCacheManager } from './KVCacheManager.ts';

/**
 * 默认模拟器配置
 */
export const DEFAULT_CONFIG: SimulatorConfig = {
  prefillBaseMs: 50,              // 基础prefill时间
  prefillMsPerToken: 0.1,        // 每token的prefill时间
  decodeBaseMs: 10,              // 基础decode时间
  decodeMsPerToken: 1.5,         // 每token的decode时间
  kvBytesPerToken: 1024,         // 每token KV大小 (1KB)
  bandwidthBytesPerMs: 1024 * 1024 * 100, // 100MB/s 带宽
  gpuMemoryBytes: 16 * 1024 * 1024 * 1024, // 16GB GPU内存
  cpuMemoryBytes: 64 * 1024 * 1024 * 1024  // 64GB CPU内存
};

/**
 * PD分离模拟器
 * 模拟Prefill-Decode分离架构中的请求处理
 */
export class PDSimulator {
  private config: SimulatorConfig;
  private random: DeterministicRandom;
  private cacheManager: KVCacheManager;
  private results: ServingResult[];
  private batchSizes: number[];  // 用于计算batch干扰

  constructor(
    config: Partial<SimulatorConfig> = {},
    cacheMaxSizeBytes: number = 512 * 1024 * 1024
  ) {
    // 合并默认配置
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.random = new DeterministicRandom(42);
    this.cacheManager = new KVCacheManager(
      cacheMaxSizeBytes,
      this.config.kvBytesPerToken,
      32  // 默认32层
    );
    this.results = [];
    this.batchSizes = [];
  }

  /**
   * 计算batch干扰因子
   * 多请求同时decode时会增加延迟
   */
  private computeBatchInterference(batchSize: number): number {
    if (batchSize <= 1) return 1.0;
    // 简化的干扰模型：线性增长但有上限
    return Math.min(2.0, 1.0 + (batchSize - 1) * 0.05);
  }

  /**
   * 计算KV大小
   * @param tokenCount token数量
   * @param totalLayers 层数
   * @param retentionRatio 保留比例
   */
  private computeKVSize(
    tokenCount: number, 
    totalLayers: number = 32,
    retentionRatio: number = 1.0
  ): number {
    return tokenCount * this.config.kvBytesPerToken * totalLayers * retentionRatio;
  }

  /**
   * 计算KV传输时间
   * @param kvSize KV大小
   * @param compression 压缩配置
   */
  private computeKVTransferTime(
    kvSize: number,
    compression: CompressionConfig | null
  ): number {
    if (!compression || compression.strategy === 'none') {
      // 无压缩：传输完整KV
      return kvSize / this.config.bandwidthBytesPerMs;
    }
    
    // 有压缩：传输压缩后的KV
    const avgRetention = computeAverageRetention(compression);
    const compressedSize = kvSize * avgRetention;
    return compressedSize / this.config.bandwidthBytesPerMs;
  }

  /**
   * 计算Prefill时间
   * @param inputTokens 输入token数
   * @param cacheHitRatio 缓存命中率
   */
  private computePrefillTime(
    inputTokens: number,
    cacheHitRatio: number = 0
  ): number {
    // 缓存命中时，跳过部分prefill
    const effectiveTokens = inputTokens * (1 - cacheHitRatio);
    return this.config.prefillBaseMs + effectiveTokens * this.config.prefillMsPerToken;
  }

  /**
   * 计算Decode时间
   * @param outputTokens 输出token数
   * @param batchSize 当前batch大小
   */
  private computeDecodeTime(
    outputTokens: number,
    batchSize: number
  ): number {
    const interference = this.computeBatchInterference(batchSize);
    return this.config.decodeBaseMs + 
           outputTokens * this.config.decodeMsPerToken * interference;
  }

  /**
   * 模拟单个请求
   * @param request 请求
   * @param compression 压缩配置
   * @param currentBatchSize 当前batch大小
   */
  simulateRequest(
    request: ServingRequest,
    compression: CompressionConfig | null = null,
    currentBatchSize: number = 1
  ): ServingResult {
    const { id, inputTokens, outputTokens, taskType, prefixHash, arrivalTimeMs } = request;
    
    // 检查缓存命中
    let cacheHit = false;
    let savedTokens = 0;
    if (prefixHash) {
      const cacheResult = this.cacheManager.lookup(prefixHash);
      cacheHit = cacheResult.hit;
      savedTokens = cacheResult.savedTokens;
    }
    
    // 计算缓存命中率
    const cacheHitRatio = cacheHit ? Math.min(savedTokens / inputTokens, 1) : 0;
    
    // 计算prefill时间（可能跳过部分）
    const prefillTime = this.computePrefillTime(inputTokens, cacheHitRatio);
    
    // 计算KV大小和传输时间
    const avgRetention = compression ? computeAverageRetention(compression) : 1.0;
    const totalLayers = compression?.pLayers?.[0]?.totalLayers || 32;
    const kvSize = this.computeKVSize(inputTokens, totalLayers, 1.0); // 原始大小
    const kvTransferTime = this.computeKVTransferTime(kvSize, compression);
    
    // TTFT = prefill + KV传输
    const ttft = prefillTime + kvTransferTime;
    
    // TPOT (Time Per Output Token)
    const tpot = this.computeDecodeTime(1, currentBatchSize);
    
    // E2E延迟
    const decodeTime = this.computeDecodeTime(outputTokens, currentBatchSize);
    const e2eLatency = ttft + decodeTime;
    
    // 质量分数
    const qualityScore = computeQualityScore(compression, taskType);
    
    // 压缩比
    const compressionRatio = compression ? avgRetention : 1.0;
    
    // 存储到缓存（如果需要）
    if (prefixHash && !cacheHit) {
      this.cacheManager.store(prefixHash, inputTokens);
    }
    
    const result: ServingResult = {
      requestId: id,
      ttftMs: MathUtils.round(ttft, 4),
      tpotMs: MathUtils.round(tpot, 4),
      e2eLatencyMs: MathUtils.round(e2eLatency, 4),
      kvTransferTimeMs: MathUtils.round(kvTransferTime, 4),
      compressionRatio: MathUtils.round(compressionRatio, 4),
      qualityScore: MathUtils.round(qualityScore, 4),
      cacheHit,
      taskType
    };
    
    this.results.push(result);
    return result;
  }

  /**
   * 模拟批量请求
   * @param requests 请求列表
   * @param compression 压缩配置
   */
  simulateBatch(
    requests: ServingRequest[],
    compression: CompressionConfig | null = null
  ): SimulationStats {
    this.results = [];
    this.batchSizes = [];
    
    // 按到达时间排序
    const sortedRequests = [...requests].sort(
      (a, b) => a.arrivalTimeMs - b.arrivalTimeMs
    );
    
    // 模拟每个请求
    for (let i = 0; i < sortedRequests.length; i++) {
      const currentBatchSize = Math.min(i + 1, 8); // 简化的batch模型
      this.simulateRequest(sortedRequests[i], compression, currentBatchSize);
      this.batchSizes.push(currentBatchSize);
    }
    
    return this.computeStats();
  }

  /**
   * 计算统计信息
   */
  computeStats(): SimulationStats {
    if (this.results.length === 0) {
      return this.createEmptyStats();
    }
    
    const ttfts = this.results.map(r => r.ttftMs);
    const tpots = this.results.map(r => r.tpotMs);
    const e2es = this.results.map(r => r.e2eLatencyMs);
    const qualities = this.results.map(r => r.qualityScore);
    const ratios = this.results.map(r => r.compressionRatio);
    const cacheHits = this.results.filter(r => r.cacheHit).length;
    
    // 按任务类型分组统计
    const taskGroups: Record<TaskType, ServingResult[]> = {
      math: [],
      code: [],
      qa: [],
      conversation: [],
      unknown: []
    };
    
    for (const result of this.results) {
      taskGroups[result.taskType].push(result);
    }
    
    const perTaskStats: SimulationStats['perTaskStats'] = {
      math: this.computeTaskStats(taskGroups.math),
      code: this.computeTaskStats(taskGroups.code),
      qa: this.computeTaskStats(taskGroups.qa),
      conversation: this.computeTaskStats(taskGroups.conversation),
      unknown: this.computeTaskStats(taskGroups.unknown)
    };
    
    // 计算吞吐量 (tokens/sec)
    const totalOutputTokens = this.results.reduce(
      (sum, r) => sum + (r.e2eLatencyMs / 1000), 0
    );
    const throughput = totalOutputTokens > 0 ? 
      this.results.length / (totalOutputTokens / 1000) : 0;
    
    return {
      totalRequests: this.results.length,
      avgTTFT: MathUtils.round(MathUtils.average(ttfts), 4),
      avgTPOT: MathUtils.round(MathUtils.average(tpots), 4),
      avgE2E: MathUtils.round(MathUtils.average(e2es), 4),
      p50TTFT: MathUtils.round(MathUtils.percentile(ttfts, 50), 4),
      p95TTFT: MathUtils.round(MathUtils.percentile(ttfts, 95), 4),
      p99TTFT: MathUtils.round(MathUtils.percentile(ttfts, 99), 4),
      avgCompressionRatio: MathUtils.round(MathUtils.average(ratios), 4),
      avgQualityScore: MathUtils.round(MathUtils.average(qualities), 4),
      cacheHitRate: MathUtils.round(cacheHits / this.results.length, 4),
      throughputTokensPerSec: MathUtils.round(throughput, 4),
      perTaskStats
    };
  }

  /**
   * 计算单个任务类型的统计
   */
  private computeTaskStats(results: ServingResult[]): {
    count: number;
    avgTTFT: number;
    avgE2E: number;
    avgQuality: number;
  } {
    if (results.length === 0) {
      return { count: 0, avgTTFT: 0, avgE2E: 0, avgQuality: 0 };
    }
    
    return {
      count: results.length,
      avgTTFT: MathUtils.round(
        MathUtils.average(results.map(r => r.ttftMs)), 4
      ),
      avgE2E: MathUtils.round(
        MathUtils.average(results.map(r => r.e2eLatencyMs)), 4
      ),
      avgQuality: MathUtils.round(
        MathUtils.average(results.map(r => r.qualityScore)), 4
      )
    };
  }

  /**
   * 创建空统计对象
   */
  private createEmptyStats(): SimulationStats {
    return {
      totalRequests: 0,
      avgTTFT: 0,
      avgTPOT: 0,
      avgE2E: 0,
      p50TTFT: 0,
      p95TTFT: 0,
      p99TTFT: 0,
      avgCompressionRatio: 1,
      avgQualityScore: 1,
      cacheHitRate: 0,
      throughputTokensPerSec: 0,
      perTaskStats: {
        math: { count: 0, avgTTFT: 0, avgE2E: 0, avgQuality: 0 },
        code: { count: 0, avgTTFT: 0, avgE2E: 0, avgQuality: 0 },
        qa: { count: 0, avgTTFT: 0, avgE2E: 0, avgQuality: 0 },
        conversation: { count: 0, avgTTFT: 0, avgE2E: 0, avgQuality: 0 },
        unknown: { count: 0, avgTTFT: 0, avgE2E: 0, avgQuality: 0 }
      }
    };
  }

  /**
   * 获取缓存管理器
   */
  getCacheManager(): KVCacheManager {
    return this.cacheManager;
  }

  /**
   * 获取配置
   */
  getConfig(): SimulatorConfig {
    return { ...this.config };
  }

  /**
   * 重置模拟器状态
   */
  reset(): void {
    this.results = [];
    this.batchSizes = [];
    this.cacheManager.clear();
  }

  /**
   * 设置随机种子
   */
  setSeed(seed: number): void {
    this.random.reset(seed);
  }

  /**
   * 获取所有结果
   */
  getResults(): ServingResult[] {
    return [...this.results];
  }
}
