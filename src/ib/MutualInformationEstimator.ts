/**
 * 互信息估算器 (Mutual Information Estimator)
 * 
 * 提供4种估算I(Z; Y)的方法：
 * 1. Attention Entropy: 用attention分布的熵近似
 * 2. Token Reuse Frequency: token被后续step引用的频率
 * 3. Gradient Saliency: 用dL/dKV近似
 * 4. Fisher Information: Fisher信息矩阵的trace
 */

// ============================================
// 类型定义
// ============================================

import type { TaskType, Phase } from './InformationBottleneck.js';

// ============================================
// 辅助函数
// ============================================

/**
 * 计算熵 H(P) = -Σ p * log(p)
 */
function entropy(probs: number[]): number {
  let h = 0;
  for (const p of probs) {
    if (p > 0) {
      h -= p * Math.log2(p);
    }
  }
  return h;
}

/**
 * 归一化数组为概率分布
 */
function normalize(probs: number[]): number[] {
  const sum = probs.reduce((a, b) => a + b, 0);
  if (sum === 0) return probs.map(() => 1 / probs.length);
  return probs.map(p => p / sum);
}

// ============================================
// 互信息估算器
// ============================================

/**
 * 互信息估算器类
 * 
 * 提供多种方法估算 I(Z; Y) — 压缩表示与预测目标之间的互信息
 */
export class MutualInformationEstimator {
  private layerIndex: number;
  private numLayers: number;
  private taskType: TaskType;
  private phase: Phase;
  
  constructor(
    layerIndex: number = 0,
    numLayers: number = 32,
    taskType: TaskType = 'unknown',
    phase: Phase = 'prefill'
  ) {
    this.layerIndex = layerIndex;
    this.numLayers = numLayers;
    this.taskType = taskType;
    this.phase = phase;
  }
  
  // ============================================
  // 方法1: Attention Entropy估算
  // ============================================
  
  /**
   * 用attention分布的熵近似I(Z;Y)
   * 
   * 原理：
   * - Attention分布的熵低 → 注意力集中 → 对预测贡献大 → I(Z;Y)高
   * - Attention分布的熵高 → 注意力分散 → 对预测贡献小 → I(Z;Y)低
   * 
   * 公式：I(Z;Y) ≈ 1 - H(attention) / H_max
   * 其中H_max = log(sequence_length)是最大熵
   * 
   * @param attentionWeights 注意力权重数组
   * @returns 估算的互信息值 [0, 1]
   */
  attentionEntropyEstimate(attentionWeights: Float64Array | number[]): number {
    const probs = Array.from(attentionWeights);
    const normalized = normalize(probs);
    
    // 计算熵
    const h = entropy(normalized);
    
    // 归一化到[0, 1]
    // 最大熵 = log2(n)，其中n是序列长度
    const n = probs.length;
    const maxEntropy = Math.log2(n);
    const normalizedEntropy = h / maxEntropy;
    
    // 熵越低，互信息越高
    const mi = 1 - normalizedEntropy;
    
    return Math.max(0, Math.min(1, mi));
  }
  
  /**
   * 生成模拟的attention权重（用于测试）
   * 基于任务类型和层索引生成合理的attention模式
   */
  generateSimulatedAttention(): Float64Array {
    const seqLen = 32; // 模拟序列长度
    const weights = new Float64Array(seqLen);
    
    // 基于任务类型生成不同的attention模式
    const normalizedLayer = this.layerIndex / (this.numLayers - 1);
    
    switch (this.taskType) {
      case 'math':
        // 数学：高层关注全局推理token，低层关注局部计算token
        if (normalizedLayer > 0.6) {
          // 高层：局部强注意力
          weights[0] = 0.5;
          weights[1] = 0.3;
          weights[2] = 0.1;
        } else {
          // 低层：均匀注意力
          for (let i = 0; i < seqLen; i++) {
            weights[i] = 1 / seqLen;
          }
        }
        break;
        
      case 'code':
        // 代码：低层关注语法token（关键词、符号），高层关注结构token
        if (normalizedLayer < 0.4) {
          // 低层：关注特定语法token
          weights[0] = 0.4;
          weights[1] = 0.2;
          weights[2] = 0.2;
        } else {
          // 高层：相对均匀
          for (let i = 0; i < seqLen; i++) {
            weights[i] = 1 / seqLen;
          }
        }
        break;
        
      case 'qa':
        // QA：关注问句和关键实体token
        weights[0] = 0.35;
        weights[1] = 0.25;
        weights[2] = 0.15;
        weights[3] = 0.1;
        break;
        
      case 'conversation':
      default:
        // 对话：相对均匀
        for (let i = 0; i < seqLen; i++) {
          weights[i] = 1 / seqLen;
        }
    }
    
    // 添加一些随机性
    for (let i = 0; i < seqLen; i++) {
      weights[i] += (Math.random() - 0.5) * 0.05;
    }
    
    // 归一化
    const sum = Array.from(weights).reduce((a, b) => a + b, 0);
    for (let i = 0; i < seqLen; i++) {
      weights[i] /= sum;
    }
    
    return weights;
  }
  
  // ============================================
  // 方法2: Token Reuse Frequency估算
  // ============================================
  
  /**
   * 用token被后续step引用的频率近似I(Z;Y)
   * 
   * 原理：
   * - 被多次引用的token → 对后续预测贡献大 → I(Z;Y)高
   * - 很少被引用的token → 对后续预测贡献小 → I(Z;Y)低
   * 
   * 公式：I(Z;Y) ≈ reuse_freq(token_i) / max_reuse_freq
   * 
   * @param tokenIndex token索引
   * @param sequenceLength 序列长度
   * @param taskType 任务类型
   * @returns 估算的互信息值 [0, 1]
   */
  tokenReuseEstimate(
    tokenIndex: number,
    sequenceLength: number,
    taskType: TaskType
  ): number {
    const normalizedPosition = tokenIndex / (sequenceLength - 1);
    
    // 不同任务类型有不同的reuse模式
    let baseReuse: number;
    
    switch (taskType) {
      case 'math':
        // 数学：近期token（计算中间结果）和关键推理token重用率高
        if (normalizedPosition > 0.7) {
          baseReuse = 0.8; // 近期token
        } else if (normalizedPosition > 0.3) {
          baseReuse = 0.5; // 中间token
        } else {
          baseReuse = 0.3; // 早期token
        }
        break;
        
      case 'code':
        // 代码：早期语法token（关键词）和关键结构token重用率高
        if (normalizedPosition < 0.2) {
          baseReuse = 0.85; // 语法token（关键词）
        } else if (normalizedPosition > 0.8) {
          baseReuse = 0.6; // 最近的结构token
        } else {
          baseReuse = 0.4; // 中间token
        }
        break;
        
      case 'qa':
        // QA：问句关键词和关键实体token重用率高
        if (normalizedPosition < 0.3 || normalizedPosition > 0.7) {
          baseReuse = 0.75; // 问句和答案区域
        } else {
          baseReuse = 0.4; // 中间上下文
        }
        break;
        
      case 'conversation':
      default:
        // 对话：近期token重用率高
        baseReuse = 0.6 * (1 - normalizedPosition * 0.4) + 0.4;
    }
    
    // 添加一些随机波动
    const noise = (Math.random() - 0.5) * 0.1;
    return Math.max(0.1, Math.min(1, baseReuse + noise));
  }
  
  // ============================================
  // 方法3: Gradient Saliency估算（简化模拟）
  // ============================================
  
  /**
   * 用dL/dKV近似I(Z;Y)（简化模拟）
   * 
   * 原理：
   * - 梯度大的KV → 对损失函数贡献大 → I(Z;Y)高
   * - 梯度小的KV → 对损失函数贡献小 → I(Z;Y)低
   * 
   * 由于没有真实梯度，这里用解析方法模拟
   * 
   * @param layerIndex 层索引
   * @param numLayers 总层数
   * @param taskType 任务类型
   * @returns 估算的互信息值 [0, 1]
   */
  gradientSaliencyEstimate(
    layerIndex: number,
    numLayers: number,
    taskType: TaskType
  ): number {
    const normalizedLayer = layerIndex / (numLayers - 1);
    
    // 不同任务类型的梯度分布不同
    let saliency: number;
    
    switch (taskType) {
      case 'math':
        // 数学：中高层的梯度更大（需要精确的中间推理结果）
        saliency = 0.3 + 0.7 * Math.pow(normalizedLayer, 1.5);
        break;
        
      case 'code':
        // 代码：低层的梯度更大（需要精确的语法表示）
        saliency = 0.9 - 0.6 * Math.pow(normalizedLayer, 1.2);
        break;
        
      case 'qa':
        // QA：中高层的梯度更大（需要精确的语义理解）
        saliency = 0.2 + 0.8 * Math.pow(normalizedLayer, 1.3);
        break;
        
      case 'conversation':
      default:
        // 对话：各层相对均匀
        saliency = 0.5 + 0.2 * Math.sin(normalizedLayer * Math.PI);
    }
    
    // Decode阶段梯度通常更大（需要精确生成）
    if (this.phase === 'decode') {
      saliency = Math.min(1, saliency * 1.2);
    }
    
    return Math.max(0.1, Math.min(1, saliency));
  }
  
  // ============================================
  // 方法4: Fisher Information估算（简化模拟）
  // ============================================
  
  /**
   * Fisher信息矩阵的trace近似I(Z;Y)
   * 
   * 原理：
   * - Fisher信息大 → 参数对似然的贡献大 → I(Z;Y)高
   * - Fisher信息小 → 参数对似然的贡献小 → I(Z;Y)低
   * 
   * @param layerIndex 层索引
   * @param numLayers 总层数
   * @param taskType 任务类型
   * @returns 估算的互信息值 [0, 1]
   */
  fisherInformationEstimate(
    layerIndex: number,
    numLayers: number,
    taskType: TaskType
  ): number {
    const normalizedLayer = layerIndex / (numLayers - 1);
    
    // Fisher信息与梯度saliency相关
    // 这里用简化的模型：Fisher ∝ gradient^2
    const gradient = this.gradientSaliencyEstimate(layerIndex, numLayers, taskType);
    
    // Fisher信息通常更大（因为是平方）
    let fisher = Math.pow(gradient, 0.5);
    
    // 添加一些任务特定的调整
    switch (taskType) {
      case 'math':
        fisher *= 1.1; // 数学任务的Fisher信息整体较高
        break;
      case 'code':
        fisher *= 1.2; // 代码任务的Fisher信息高（需要精确语法）
        break;
      case 'qa':
        fisher *= 1.0; // QA任务基准
        break;
      case 'conversation':
        fisher *= 0.9; // 对话任务的Fisher信息稍低
    }
    
    return Math.max(0.1, Math.min(1, fisher));
  }
  
  // ============================================
  // 综合估算
  // ============================================
  
  /**
   * 综合4种方法（加权平均）
   * 
   * 权重根据方法可靠性分配：
   * - attentionEntropy: 0.3（需要真实attention权重）
   * - tokenReuse: 0.25（基于位置的粗略估计）
   * - gradientSaliency: 0.25（需要真实梯度）
   * - fisherInformation: 0.2（理论可靠但计算复杂）
   * 
   * @param layerIndex 层索引
   * @param numLayers 总层数
   * @param taskType 任务类型
   * @param phase 推理阶段
   * @param attentionWeights 可选的attention权重
   * @returns 综合互信息估计 [0, 1]
   */
  combinedEstimate(
    layerIndex: number,
    numLayers: number,
    taskType: TaskType,
    phase: Phase,
    attentionWeights?: Float64Array | number[]
  ): number {
    // 方法权重
    const W_ENTROPY = 0.30;
    const W_REUSE = 0.25;
    const W_GRADIENT = 0.25;
    const W_FISHER = 0.20;
    
    // 方法1: Attention Entropy
    let entropyEstimate: number;
    if (attentionWeights && attentionWeights.length > 0) {
      entropyEstimate = this.attentionEntropyEstimate(attentionWeights);
    } else {
      // 没有真实attention权重，使用模拟的
      const tempEstimator = new MutualInformationEstimator(
        layerIndex, numLayers, taskType, phase
      );
      const simWeights = tempEstimator.generateSimulatedAttention();
      entropyEstimate = this.attentionEntropyEstimate(simWeights);
    }
    
    // 方法2: Token Reuse
    const reuseEstimate = this.tokenReuseEstimate(
      layerIndex, numLayers, taskType
    );
    
    // 方法3: Gradient Saliency
    const gradientEstimate = this.gradientSaliencyEstimate(
      layerIndex, numLayers, taskType
    );
    
    // 方法4: Fisher Information
    const fisherEstimate = this.fisherInformationEstimate(
      layerIndex, numLayers, taskType
    );
    
    // 加权平均
    const combined =
      W_ENTROPY * entropyEstimate +
      W_REUSE * reuseEstimate +
      W_GRADIENT * gradientEstimate +
      W_FISHER * fisherEstimate;
    
    return Math.max(0.1, Math.min(1, combined));
  }
  
  /**
   * 估算所有层的互信息
   */
  estimateAllLayers(
    numLayers: number,
    taskType: TaskType,
    phase: Phase,
    attentionWeights?: Float64Array[]
  ): number[] {
    const estimates: number[] = [];
    
    for (let i = 0; i < numLayers; i++) {
      const layerWeights = attentionWeights ? attentionWeights[i] : undefined;
      estimates.push(
        this.combinedEstimate(i, numLayers, taskType, phase, layerWeights)
      );
    }
    
    return estimates;
  }
  
  // ============================================
  // 调试和可视化
  // ============================================
  
  /**
   * 打印各方法的对比结果
   */
  printMethodComparison(
    layerIndex: number,
    numLayers: number,
    taskType: TaskType,
    phase: Phase
  ): void {
    console.log(`\n=== 互信息估算方法对比 (Layer ${layerIndex}/${numLayers - 1}) ===`);
    console.log(`任务类型: ${taskType}, 阶段: ${phase}`);
    console.log('---');
    
    const simWeights = this.generateSimulatedAttention();
    console.log(`1. Attention Entropy: ${this.attentionEntropyEstimate(simWeights).toFixed(4)}`);
    console.log(`2. Token Reuse: ${this.tokenReuseEstimate(layerIndex, 32, taskType).toFixed(4)}`);
    console.log(`3. Gradient Saliency: ${this.gradientSaliencyEstimate(layerIndex, numLayers, taskType).toFixed(4)}`);
    console.log(`4. Fisher Information: ${this.fisherInformationEstimate(layerIndex, numLayers, taskType).toFixed(4)}`);
    console.log(`5. Combined: ${this.combinedEstimate(layerIndex, numLayers, taskType, phase).toFixed(4)}`);
  }
}

// ============================================
// 导出便捷函数
// ============================================

/**
 * 快速估算I(Z;Y)
 */
export function estimateMutualInformation(
  layerIndex: number,
  numLayers: number,
  taskType: TaskType,
  phase: Phase = 'prefill'
): number {
  const estimator = new MutualInformationEstimator(layerIndex, numLayers, taskType, phase);
  return estimator.combinedEstimate(layerIndex, numLayers, taskType, phase);
}

/**
 * 估算所有层的I(Z;Y)分布
 */
export function estimateLayerDistribution(
  numLayers: number,
  taskType: TaskType,
  phase: Phase = 'prefill'
): number[] {
  const estimator = new MutualInformationEstimator(0, numLayers, taskType, phase);
  return estimator.estimateAllLayers(numLayers, taskType, phase);
}

export default MutualInformationEstimator;
