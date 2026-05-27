// ============================================
// 统一类型系统 - kvcache-lab
// 权威类型来源，所有模块应从本文件导入类型
// ============================================

// ============================================
// 任务类型定义
// ============================================
export type TaskType = 'math' | 'code' | 'qa' | 'conversation' | 'unknown';

// 分类结果接口
export interface ClassificationResult {
  taskType: TaskType;
  confidence: number;
  method: 'rule' | 'api';
  latencyMs: number;
}

// 分类器选项
export interface TaskClassifierOptions {
  useAPI?: boolean;
  apiTimeout?: number;
}

// ============================================
// 压缩相关类型
// ============================================

// 压缩策略类型
export type CompressionStrategyType = 'none' | 'uniform' | 'pd-aware' | 'task-aware';

// 层级压缩配置
export interface LayerCompressionConfig {
  layerIndex: number;
  totalLayers: number;
  retentionRatio: number;  // 0-1, 保留的KV比例
  keyPrecision: number;    // bits per key element
  valuePrecision: number;  // bits per value element
}

// 压缩配置
export interface CompressionConfig {
  strategy: CompressionStrategyType;
  pLayers: LayerCompressionConfig[];   // P端（传输前）的层配置
  dLayers: LayerCompressionConfig[];   // D端（接收后）的层配置
  taskType?: TaskType;                 // 任务类型（task-aware用）
}

// 压缩结果
export interface CompressionResult {
  originalSizeBytes: number;
  compressedSizeBytes: number;
  compressionRatio: number;          // compressed/original
  perLayerRetention: number[];       // 每层保留比例
  estimatedQualityImpact: number;    // 0-1, 预估质量影响
}

// KV传输结果
export interface KVTransferResult {
  transferTimeMs: number;
  effectiveTTFTOverheadMs: number;
  bytesTransferred: number;
  compressionApplied: CompressionConfig | null;
}

// 压缩参数输入（压缩策略计算用）
export interface CompressionParams {
  totalLayers: number;           // 模型总层数（如32层）
  totalTokens: number;           // 总token数
  bandwidthBytesPerMs: number;   // 网络带宽 (bytes/ms)
  gpuMemoryBytes: number;        // GPU显存总量
  currentMemoryUsage: number;    // 当前显存使用量
  taskType: string;              // 任务类型: 'math' | 'code' | 'qa' | 'conversation'
  sloLatencyMs?: number;         // SLO约束 (ms)
  prefixHitRate?: number;        // 前缀命中率 [0-1]
}

// 压缩配置输出（压缩策略返回用）
export interface CompressionOutput {
  strategy: string;
  totalLayers: number;
  pLayerRetention: number[];     // P端每层保留比例 [0.1-1.0]
  dLayerRetention: number[];     // D端每层保留比例 [0.1-1.0]
  pKeyPrecision: number[];       // P端每层Key精度(bits)
  pValuePrecision: number[];     // P端每层Value精度(bits)
  dKeyPrecision: number[];       // D端每层Key精度(bits)
  dValuePrecision: number[];     // D端每层Value精度(bits)
  avgCompressionRatio: number;   // 平均压缩比
  estimatedBandwidthSaving: number; // 预估带宽节省比例 [0-1]
}

// 压缩策略接口
export interface ICompressionStrategy {
  readonly name: string;
  readonly type: string;  // 'none' | 'uniform' | 'pd-aware' | 'task-aware'
  
  // 计算压缩配置
  computeConfig(params: CompressionParams): CompressionOutput;
  
  // 预估质量影响 (0-1, 越高越好)
  estimateQualityImpact(config: CompressionOutput, taskType: string): number;
}

// ============================================
// 请求与响应类型
// ============================================

// 请求
export interface ServingRequest {
  id: string;
  inputTokens: number;
  outputTokens: number;
  taskType: TaskType;
  prefixHash?: string;    // 前缀hash，用于缓存命中
  arrivalTimeMs: number;
  sloLatencyMs?: number;  // SLO延迟约束
}

// 请求结果
export interface ServingResult {
  requestId: string;
  ttftMs: number;
  tpotMs: number;
  e2eLatencyMs: number;
  kvTransferTimeMs: number;
  compressionRatio: number;
  qualityScore: number;    // 0-1
  cacheHit: boolean;
  taskType: TaskType;
}

// ============================================
// 模拟器类型
// ============================================

// 模拟器配置
export interface SimulatorConfig {
  prefillBaseMs: number;
  prefillMsPerToken: number;
  decodeBaseMs: number;
  decodeMsPerToken: number;
  kvBytesPerToken: number;      // 每token KV大小(bytes)
  bandwidthBytesPerMs: number;  // 网络带宽
  gpuMemoryBytes: number;
  cpuMemoryBytes: number;
}

// 模拟器统计
export interface SimulationStats {
  totalRequests: number;
  avgTTFT: number;
  avgTPOT: number;
  avgE2E: number;
  p50TTFT: number;
  p95TTFT: number;
  p99TTFT: number;
  avgCompressionRatio: number;
  avgQualityScore: number;
  cacheHitRate: number;
  throughputTokensPerSec: number;
  perTaskStats: Record<TaskType, {
    count: number;
    avgTTFT: number;
    avgE2E: number;
    avgQuality: number;
  }>;
}

// ============================================
// 层预算分配类型
// ============================================

// 精度类型
export type PrecisionType = 'FP16' | 'FP8' | 'INT4';

// 层预算接口
export interface LayerBudget {
  totalLayers: number;
  retentionRatios: number[];      // 每层KV保留比例
  keyPrecisions: number[];        // 每层Key精度(bits)
  valuePrecisions: number[];     // 每层Value精度(bits)
  totalBudgetBytes: number;       // 总KV预算(bytes)
  perLayerBudgetBytes: number[];  // 每层预算
}

// 预算约束接口
export interface BudgetConstraints {
  totalMemoryBytes: number;    // 可用总显存
  totalLayers: number;         // 模型层数
  hiddenSize: number;          // 隐藏维度
  numHeads: number;             // 注意力头数
  sequenceLength: number;       // 序列长度
  taskType: TaskType;
  sloLatencyMs?: number;       // SLO延迟目标(毫秒)
}

// 任务Profile类型
export type ProfileType = 'inverted' | 'normal' | 'flat';

// ============================================
// 工具函数
// ============================================

// 限制值在[min, max]范围内
export function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

// 保留4位小数精度
export function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

// 确保保留率在[0.1, 1.0]范围内
export function ensureRetentionRange(value: number): number {
  return clamp(round4(value), 0.1, 1.0);
}
