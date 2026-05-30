/**
 * 压缩配置适配器
 * 将 CompressionOrchestrator 的 CompressionOutput 转换为 PDSimulator 的 CompressionConfig
 * 
 * 问题：CompressionOrchestrator 输出 per-layer retention + precision，
 *       PDSimulator 期望 CompressionConfig { strategy, pLayers, dLayers }
 * 解决：本模块完成两者的桥接
 */

import {
  CompressionOutput,
  CompressionConfig,
  LayerCompressionConfig,
} from '../core/types.js';

/**
 * 将 CompressionOutput 转为 PDSimulator 可用的 CompressionConfig
 */
export function adaptCompressionOutput(
  output: CompressionOutput,
  taskType?: string,
): CompressionConfig {
  const totalLayers = output.totalLayers;

  const pLayers: LayerCompressionConfig[] = output.pLayerRetention.map((ret, i) => ({
    layerIndex: i,
    totalLayers,
    retentionRatio: ret,
    keyPrecision: output.pKeyPrecision[i] ?? 16,
    valuePrecision: output.pValuePrecision[i] ?? 16,
  }));

  const dLayers: LayerCompressionConfig[] = output.dLayerRetention.map((ret, i) => ({
    layerIndex: i,
    totalLayers,
    retentionRatio: ret,
    keyPrecision: output.dKeyPrecision[i] ?? 16,
    valuePrecision: output.dValuePrecision[i] ?? 16,
  }));

  // 映射 strategy 名到 type
  const strategyMap: Record<string, CompressionConfig['strategy']> = {
    'NoneCompression': 'none',
    'UniformCompression': 'uniform',
    'PDAwareCompression': 'pd-aware',
    'TaskAwareCompression': 'task-aware',
    'PDTaskAwareCompression': 'pd-aware',  // 联合策略使用pd-aware type
    'QualityConstrainedCompression': 'uniform',
  };

  return {
    strategy: strategyMap[output.strategy] ?? 'uniform',
    pLayers,
    dLayers,
    taskType: taskType as any,
  };
}

/**
 * 批量适配：对多种策略参数组合生成 CompressionConfig[]
 */
export function batchAdapt(
  outputs: { output: CompressionOutput; taskType?: string }[],
): CompressionConfig[] {
  return outputs.map(({ output, taskType }) => adaptCompressionOutput(output, taskType));
}
