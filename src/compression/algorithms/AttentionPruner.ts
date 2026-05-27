/**
 * 注意力分数驱动的Token剪枝算法
 * 
 * 基于attention分数的token重要性评估，实现智能剪枝
 * 策略：
 * - head_tail: 首尾token保留策略（PDTrim风格）
 * - importance: 基于模拟attention分数的重要性剪枝
 * - random: 随机剪枝作为baseline
 */

import { MathUtils } from '../../core/utils.js';

/**
 * 剪枝策略类型
 */
export type PruningStrategy = 'head_tail' | 'importance' | 'random';

/**
 * 剪枝配置
 */
export interface PruningConfig {
  retentionRatio: number;        // 保留比例 [0, 1]
  strategy: PruningStrategy;      // 剪枝策略
  headRetentionRatio: number;   // 头部保留比例
  tailRetentionRatio: number;    // 尾部保留比例
}

/**
 * 剪枝结果
 */
export interface PruningResult {
  retainedIndices: number[];    // 保留的token索引
  prunedIndices: number[];       // 被剪枝的token索引
  compressionRatio: number;     // 压缩比例（保留比例）
  estimatedImpact: number;      // 预估质量影响
}

/**
 * 注意力剪枝器
 */
export class AttentionPruner {
  /**
   * 生成模拟的attention分数
   * 
   * 不同任务类型的attention分布：
   * - math: 中间token分数较高（推理步骤重要）
   * - code: 低层token分数较高（语法结构重要）
   * - qa: 头部token分数较高（system prompt + question）
   * - conversation: 均匀分布
   * 
   * @param tokenCount token数量
   * @param taskType 任务类型
   * @param seed 随机种子（用于确定性）
   */
  generateSimulatedAttentionScores(
    tokenCount: number,
    taskType: string,
    seed: number = 42
  ): Float64Array {
    const scores = new Float64Array(tokenCount);
    const rng = new DeterministicRandom(seed);
    
    // 根据任务类型生成不同的attention分布
    for (let i = 0; i < tokenCount; i++) {
      const position = i / (tokenCount - 1); // 归一化位置 [0, 1]
      
      let baseScore: number;
      
      switch (taskType.toLowerCase()) {
        case 'math':
          // 数学任务：中间位置（推理步骤）更重要
          // 使用二次函数，中间高，两端低
          baseScore = 1 - Math.pow(2 * position - 1, 2);
          break;
          
        case 'code':
          // 代码任务：语法结构重要，低层token更关键
          // 指数衰减，头部高
          baseScore = Math.exp(-3 * position);
          break;
          
        case 'qa':
          // QA任务：问题本身重要，头部token关键
          // 快速衰减
          baseScore = Math.exp(-5 * position);
          break;
          
        case 'conversation':
          // 对话：相对均匀分布
          baseScore = 0.7 + 0.3 * (1 - Math.abs(2 * position - 1));
          break;
          
        default:
          // 默认：接近均匀
          baseScore = 0.5 + 0.5 * (1 - Math.abs(2 * position - 1));
      }
      
      // 添加一些噪声模拟真实attention的随机性
      const noise = (rng.next() - 0.5) * 0.2 * baseScore;
      scores[i] = Math.max(0, Math.min(1, baseScore + noise));
    }
    
    return scores;
  }
  
  /**
   * 基于attention分数的剪枝
   * 
   * 保留分数最高的token，剪枝分数较低的token
   */
  prune(
    attentionScores: Float64Array,
    config: PruningConfig
  ): PruningResult {
    const tokenCount = attentionScores.length;
    const targetRetain = Math.ceil(tokenCount * config.retentionRatio);
    
    // 创建索引数组并按attention分数排序
    const indexedScores: Array<{ index: number; score: number }> = [];
    for (let i = 0; i < tokenCount; i++) {
      indexedScores.push({ index: i, score: attentionScores[i] });
    }
    
    // 按分数降序排序
    indexedScores.sort((a, b) => b.score - a.score);
    
    // 选择要保留的token
    const retainedSet = new Set<number>();
    for (let i = 0; i < targetRetain; i++) {
      retainedSet.add(indexedScores[i].index);
    }
    
    // 分类保留和剪枝的token
    const retainedIndices: number[] = [];
    const prunedIndices: number[] = [];
    
    for (let i = 0; i < tokenCount; i++) {
      if (retainedSet.has(i)) {
        retainedIndices.push(i);
      } else {
        prunedIndices.push(i);
      }
    }
    
    // 估算质量影响
    const impact = this.estimatePruningImpact(
      attentionScores,
      prunedIndices,
      tokenCount
    );
    
    return {
      retainedIndices,
      prunedIndices,
      compressionRatio: MathUtils.round(config.retentionRatio, 4),
      estimatedImpact: MathUtils.round(impact, 4)
    };
  }
  
  /**
   * 首尾保留策略（PDTrim风格）
   * 
   * 保留头部的system prompt和尾部的recent tokens，
   * 剪枝中间的推理/生成过程
   */
  headTailPrune(
    tokenCount: number,
    config: PruningConfig
  ): PruningResult {
    const { headRetentionRatio, tailRetentionRatio } = config;
    
    // 计算各部分token数
    const headCount = Math.ceil(tokenCount * headRetentionRatio * 0.3); // 头部占30%
    const tailCount = Math.ceil(tokenCount * tailRetentionRatio * 0.3); // 尾部占30%
    
    // 确保有足够的token保留
    const minRetain = Math.max(headCount + tailCount, 
      Math.ceil(tokenCount * config.retentionRatio));
    
    // 调整尾部数量以满足总保留比例
    const availableForTail = Math.floor(tokenCount * config.retentionRatio) - headCount;
    const actualTailCount = Math.max(0, Math.min(tailCount, availableForTail));
    
    // 确定保留的索引范围
    const retainedIndices: number[] = [];
    const prunedIndices: number[] = [];
    
    for (let i = 0; i < tokenCount; i++) {
      // 头部保留
      if (i < headCount) {
        retainedIndices.push(i);
      }
      // 尾部保留（从后往前）
      else if (i >= tokenCount - actualTailCount) {
        retainedIndices.push(i);
      }
      // 中间剪枝
      else {
        prunedIndices.push(i);
      }
    }
    
    // 计算实际压缩比
    const actualRatio = retainedIndices.length / tokenCount;
    
    // 首尾策略的质量影响估算（相对较小）
    const impact = this.estimateHeadTailImpact(
      tokenCount,
      retainedIndices.length,
      headRetentionRatio,
      tailRetentionRatio
    );
    
    return {
      retainedIndices,
      prunedIndices,
      compressionRatio: MathUtils.round(actualRatio, 4),
      estimatedImpact: MathUtils.round(impact, 4)
    };
  }
  
  /**
   * 随机剪枝（baseline）
   */
  randomPrune(
    tokenCount: number,
    config: PruningConfig,
    seed: number = 42
  ): PruningResult {
    const rng = new DeterministicRandom(seed);
    const targetRetain = Math.ceil(tokenCount * config.retentionRatio);
    
    // 创建索引并随机打乱
    const indices = Array.from({ length: tokenCount }, (_, i) => i);
    const shuffled = this.shuffleArray(indices, rng);
    
    // 选择保留的token
    const retainedSet = new Set<number>();
    for (let i = 0; i < targetRetain; i++) {
      retainedSet.add(shuffled[i]);
    }
    
    const retainedIndices: number[] = [];
    const prunedIndices: number[] = [];
    
    for (let i = 0; i < tokenCount; i++) {
      if (retainedSet.has(i)) {
        retainedIndices.push(i);
      } else {
        prunedIndices.push(i);
      }
    }
    
    // 随机剪枝的质量影响通常较大（因为可能剪掉重要token）
    const impact = 1 - config.retentionRatio;
    
    return {
      retainedIndices,
      prunedIndices,
      compressionRatio: MathUtils.round(config.retentionRatio, 4),
      estimatedImpact: MathUtils.round(impact, 4)
    };
  }
  
  /**
   * 执行剪枝（根据策略选择）
   */
  executePrune(
    tokenCount: number,
    attentionScores: Float64Array | null,
    config: PruningConfig,
    taskType: string = 'unknown',
    seed: number = 42
  ): PruningResult {
    switch (config.strategy) {
      case 'head_tail':
        return this.headTailPrune(tokenCount, config);
        
      case 'importance':
        // 如果没有提供attention分数，先生成模拟分数
        const scores = attentionScores || 
          this.generateSimulatedAttentionScores(tokenCount, taskType, seed);
        return this.prune(scores, config);
        
      case 'random':
        return this.randomPrune(tokenCount, config, seed);
        
      default:
        throw new Error(`Unknown pruning strategy: ${config.strategy}`);
    }
  }
  
  /**
   * 估算剪枝对质量的影响
   * 
   * 假设被剪枝的token的平均attention分数越高，影响越大
   */
  private estimatePruningImpact(
    attentionScores: Float64Array,
    prunedIndices: number[],
    totalTokens: number
  ): number {
    if (prunedIndices.length === 0) return 0;
    
    // 计算被剪枝token的平均attention分数
    let totalScore = 0;
    for (const idx of prunedIndices) {
      totalScore += attentionScores[idx];
    }
    const avgPrunedScore = totalScore / prunedIndices.length;
    
    // 计算保留token的平均attention分数
    let retainedScore = 0;
    for (let i = 0; i < totalTokens; i++) {
      if (!prunedIndices.includes(i)) {
        retainedScore += attentionScores[i];
      }
    }
    const retainedCount = totalTokens - prunedIndices.length;
    const avgRetainedScore = retainedScore / retainedCount;
    
    // 影响 = (被剪枝分数 - 保留分数) / 被剪枝分数
    // 如果被剪枝的分数低，影响小；反之影响大
    const impact = avgRetainedScore > 0 ? 
      (avgRetainedScore - avgPrunedScore) / avgRetainedScore * 0.5 : 
      0;
    
    // 归一化到 [0, 1]
    return Math.min(1, Math.max(0, impact + (1 - totalTokens / (totalTokens - prunedIndices.length))));
  }
  
  /**
   * 估算首尾策略的质量影响
   */
  private estimateHeadTailImpact(
    totalTokens: number,
    retainedTokens: number,
    headRatio: number,
    tailRatio: number
  ): number {
    // 首尾策略的影响因子
    // 头部保留率高 + 尾部保留率高 → 影响小
    const headContribution = (1 - headRatio) * 0.2;   // 头部影响权重
    const tailContribution = (1 - tailRatio) * 0.1;   // 尾部影响权重
    
    // 压缩比例的影响
    const compressionRatio = retainedTokens / totalTokens;
    const compressionImpact = (1 - compressionRatio) * 0.3;
    
    return headContribution + tailContribution + compressionImpact;
  }
  
  /**
   * Fisher-Yates洗牌算法
   */
  private shuffleArray<T>(arr: T[], rng: DeterministicRandom): T[] {
    const result = [...arr];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(rng.next() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
  
  /**
   * 创建剪枝配置的工厂方法
   */
  static createConfig(
    strategy: PruningStrategy,
    retentionRatio: number,
    headRetentionRatio: number = 0.8,
    tailRetentionRatio: number = 0.8
  ): PruningConfig {
    return {
      strategy,
      retentionRatio,
      headRetentionRatio,
      tailRetentionRatio
    };
  }
}

/**
 * 确定性随机数生成器
 */
class DeterministicRandom {
  private state: number;

  constructor(seed: number = 42) {
    this.state = seed;
  }

  next(): number {
    let t = (this.state += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
}
