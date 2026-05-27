/**
 * 统一框架模块导出 ⭐
 * 
 * IB和R-D理论统一：
 * - IB: min I(Z;X) - β × I(Z;Y) → 最大化预测信息
 * - R-D: min D s.t. R ≤ B → 带宽最优分配
 * 
 * 核心贡献：
 * 1. PD分离场景的传输R-D优化（RDKV仅做单节点bit allocation）
 * 2. Unequal Error Protection基于IB重要性
 * 3. Semantic Distortion基于生成质量（RDKV用attention失真）
 */

// 统一框架
export {
  SemanticRDFramework,
  semanticRDFramework,
  optimizeSemanticRD,
  phaseAwareOptimize,
} from './SemanticRDFramework.js';

export type {
  SemanticRDConfig,
  SemanticRDResult,
  LayerAllocation,
  TheoreticalBounds,
  RDCurvePoint,
  PhaseType,
} from './SemanticRDFramework.js';

// IB-RD统一压缩策略
export {
  IBRDCompressor,
  ibrdCompressor,
} from './IBRDCompressor.js';

// 理论分析
export {
  TheoreticalAnalysis,
  theoreticalAnalysis,
} from './TheoreticalAnalysis.js';

export type {
  PDDifferentiationProof,
  OptimalBetaResult,
  RDBoundResult,
  TheoryValidationResult,
} from './TheoreticalAnalysis.js';
