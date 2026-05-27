/**
 * 任务感知模块 - 导出入口
 * 
 * 提供任务分类和层预算分配功能
 * 类型定义统一从 ../core/types.ts 导入
 */

// TaskClassifier 导出
export {
  TaskClassifier,
  classifyTask,
  classifyTaskBatch,
} from './TaskClassifier.js';

export type {
  ClassificationResult,
  TaskClassifierOptions,
} from '../core/types.js';

// LayerBudgetAllocator 导出
export {
  LayerBudgetAllocator,
  allocateLayerBudget,
  generatePyramid,
  getPrecisionForLayer,
  TASK_PROFILES,
} from './LayerBudgetAllocator.js';

export type {
  PrecisionType,
  LayerBudget,
  BudgetConstraints,
  ProfileType,
} from '../core/types.js';

// Profile 导出
export {
  MATH_LAYER_WEIGHTS,
  MATH_TOKEN_SENSITIVITY,
  MATH_COMPRESSION_PREFERENCE,
  MATH_TASK_FEATURES,
} from './profiles/math.js';

export {
  CODE_LAYER_WEIGHTS,
  CODE_TOKEN_SENSITIVITY,
  CODE_COMPRESSION_PREFERENCE,
  CODE_TASK_FEATURES,
  SYNTAX_PRIORITY,
} from './profiles/code.js';

export {
  QA_LAYER_WEIGHTS,
  QA_TOKEN_SENSITIVITY,
  QA_COMPRESSION_PREFERENCE,
  QA_TASK_FEATURES,
  ATTENTION_PATTERN,
} from './profiles/qa.js';
