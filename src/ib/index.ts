/**
 * Information Bottleneck 模块
 * 
 * [Reference] IB基础框架参考: CapKV (arXiv:2604.25975, 2026年4月)
 * CapKV用IB定义KV重要性并推导了I(Z;Y)的闭式解
 * 
 * [Contribution] 核心贡献: Phase-aware IB扩展
 * 将CapKV的统一IB优化扩展到Phase-aware场景
 * 证明P端和D端需要不同的β值: β_P > β_D
 * 
 * 模块结构:
 * - InformationBottleneck.ts: IB核心框架
 * - MutualInformationEstimator.ts: 互信息估算器
 * - PhaseAwareIB.ts: Phase-aware IB统一框架
 * - IBCompressor.ts: Phase-aware IB压缩策略
 */

// 导出核心类
export { InformationBottleneck, computeTheoreticalBound } from './InformationBottleneck.js';
export type { IBConfig, IBLayerResult, TaskType, Phase } from './InformationBottleneck.js';

// 导出互信息估算器
export { MutualInformationEstimator, estimateMutualInformation, estimateLayerDistribution } from './MutualInformationEstimator.js';

// 导出Phase-aware IB
export { PhaseAwareIB, optimizePhaseAwareIB } from './PhaseAwareIB.js';
export type { PhaseAwareIBConfig, PhaseAwareIBResult, RDCurvePoint } from './PhaseAwareIB.js';

// 导出压缩策略
export { PhaseAwareIBCompressor, CapKVBaselineCompressor } from './IBCompressor.js';
