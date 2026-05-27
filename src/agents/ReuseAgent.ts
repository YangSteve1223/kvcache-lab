/**
 * Memory Lifetime Agent (Reuse Agent)
 * 
 * 负责：预测KV的reuse distance和reuse probability
 * 输出：ReuseState（写入Global State Store）
 * 
 * 核心能力：
 * - 预测token的reuse distance
 * - 预测token的reuse probability
 * - 层级别的reuse预测
 * - 多种预测方法（统计/语义/任务模式）
 */

import { SemanticRegion } from './SemanticAgent.js';
import { TaskType } from '../core/types.js';

// ============================================
// 类型定义
// ============================================

/**
 * Reuse Agent 输入接口
 */
export interface ReuseAgentInput {
  tokenCount: number;
  numLayers: number;
  taskType: TaskType;
  currentStep: number;
  generationProgress: number;
  historicalAttention: Float64Array[];   // 历史attention分布
  semanticRegions: SemanticRegion[];       // 从Global State读取
}

/**
 * 单个token的reuse预测
 */
export interface TokenReusePrediction {
  reuseDistance: number;      // 预测的下次访问步数
  reuseProbability: number;   // 0-1 再访问概率
  confidence: number;         // 0-1 预测置信度
  method: 'statistical' | 'semantic' | 'task-pattern' | 'combined';
}

/**
 * 层级别的reuse预测
 */
export interface LayerReusePrediction {
  avgReuseDistance: number;
  hotTokens: number[];        // 该层热token列表
  coldTokens: number[];       // 该层冷token列表
}

/**
 * Reuse状态输出
 */
export interface ReuseState {
  tokenPredictions: Map<number, TokenReusePrediction>;
  layerPredictions: Map<number, LayerReusePrediction>;
  globalReuseRate: number;     // 全局reuse率估计
  evictableTokens: number[];   // 可驱逐的token索引
}

// ============================================
// 常量定义
// ============================================

// EMA平滑系数
const EMA_ALPHA = 0.3;

// 任务特定的reuse模式
const TASK_REUSE_PATTERNS = {
  math: {
    systemPrompt: { distance: 1, probability: 0.99 },
    recentReasoning: { distance: 3, probability: 0.85 },
    middleReasoning: { distance: 15, probability: 0.5 },
    earlyReasoning: { distance: 20, probability: 0.3 }
  },
  code: {
    currentFunction: { distance: 1, probability: 0.95 },
    currentFunctionBody: { distance: 2, probability: 0.9 },
    otherFunction: { distance: 25, probability: 0.4 },
    imports: { distance: Infinity, probability: 0.05 }
  },
  qa: {
    systemPrompt: { distance: 1, probability: 0.99 },
    currentChunk: { distance: 2, probability: 0.8 },
    relatedChunk: { distance: 5, probability: 0.5 },
    otherChunk: { distance: Infinity, probability: 0.1 }
  },
  conversation: {
    systemPrompt: { distance: 1, probability: 0.99 },
    recentTurn: { distance: 1, probability: 0.95 },
    middleTurn: { distance: 3, probability: 0.7 },
    earlyTurn: { distance: Infinity, probability: 0.2 }
  }
};

// ============================================
// ReuseAgent 类实现
// ============================================

export class ReuseAgent {
  private accessHistory: Map<number, number[]> = new Map(); // token -> 访问步骤历史
  private emaIntervals: Map<number, number> = new Map();    // token -> EMA访问间隔
  private predictionCache: Map<number, TokenReusePrediction> = new Map();
  
  /**
   * 核心方法：预测reuse
   * 
   * @param input 输入参数
   * @returns ReuseState（写入Global State Store）
   */
  predict(input: ReuseAgentInput): ReuseState {
    const { tokenCount, numLayers, taskType, currentStep, generationProgress, historicalAttention, semanticRegions } = input;
    
    // 1. 更新访问历史
    this.updateAccessHistory(historicalAttention, currentStep);
    
    // 2. 统计预测
    const statisticalPredictions = this.statisticalPredict(historicalAttention, currentStep);
    
    // 3. 语义预测
    const semanticPredictions = this.semanticPredict(tokenCount, taskType, currentStep, semanticRegions);
    
    // 4. 任务模式预测
    const taskPatternPredictions = this.taskPatternPredict(taskType, tokenCount, currentStep, generationProgress);
    
    // 5. 综合预测（加权平均）
    const combinedPredictions = this.combinePredictions(
      statisticalPredictions,
      semanticPredictions,
      taskPatternPredictions
    );
    
    // 6. 层级别预测
    const layerPredictions = this.predictLayerReuse(combinedPredictions, numLayers);
    
    // 7. 确定可驱逐token
    const evictableTokens = this.identifyEvictableTokens(combinedPredictions, semanticRegions);
    
    // 8. 计算全局reuse率
    const globalReuseRate = this.calculateGlobalReuseRate(combinedPredictions);
    
    return {
      tokenPredictions: combinedPredictions,
      layerPredictions,
      globalReuseRate,
      evictableTokens
    };
  }
  
  /**
   * 方法1：统计预测（EMA attention）
   * 
   * 算法：
   * 对每个token t:
   *   收集历史访问步骤 accessSteps
   *   如果从未被访问: reuseDistance = ∞, reuseProbability = 0
   *   否则:
   *     计算访问间隔 intervals = [accessSteps[i] - accessSteps[i-1]]
   *     EMA预测: predictedInterval = α × lastInterval + (1-α) × previousEMA
   *     reuseDistance = predictedInterval
   *     reuseProbability = 1 / (1 + predictedInterval / avgInterval)
   *     confidence = min(1, accessCount / 10)
   */
  private statisticalPredict(
    historicalAttention: Float64Array[],
    currentStep: number
  ): Map<number, TokenReusePrediction> {
    const predictions = new Map<number, TokenReusePrediction>();
    
    // 对每个token
    for (let tokenId = 0; tokenId < this.accessHistory.size; tokenId++) {
      const accessSteps = this.accessHistory.get(tokenId) || [];
      
      if (accessSteps.length === 0) {
        // 从未被访问
        predictions.set(tokenId, {
          reuseDistance: Infinity,
          reuseProbability: 0,
          confidence: 0,
          method: 'statistical'
        });
        continue;
      }
      
      // 计算访问间隔
      const intervals: number[] = [];
      for (let i = 1; i < accessSteps.length; i++) {
        intervals.push(accessSteps[i] - accessSteps[i - 1]);
      }
      
      // EMA预测
      const lastInterval = intervals[intervals.length - 1];
      const previousEMA = this.emaIntervals.get(tokenId) || lastInterval;
      const predictedInterval = EMA_ALPHA * lastInterval + (1 - EMA_ALPHA) * previousEMA;
      
      // 更新EMA
      this.emaIntervals.set(tokenId, predictedInterval);
      
      // 计算平均间隔
      const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
      
      // 计算reuse概率
      const reuseProbability = 1 / (1 + predictedInterval / Math.max(1, avgInterval));
      
      // 置信度：基于访问次数
      const confidence = Math.min(1, accessSteps.length / 10);
      
      predictions.set(tokenId, {
        reuseDistance: Math.round(predictedInterval),
        reuseProbability: Math.round(reuseProbability * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
        method: 'statistical'
      });
    }
    
    return predictions;
  }
  
  /**
   * 方法2：语义位置预测
   * 
   * 根据token在语义区域中的位置预测reuse
   */
  private semanticPredict(
    tokenCount: number,
    taskType: TaskType,
    currentStep: number,
    regions: SemanticRegion[]
  ): Map<number, TokenReusePrediction> {
    const predictions = new Map<number, TokenReusePrediction>();
    
    if (regions.length === 0) {
      // 无区域信息，返回均匀分布
      for (let i = 0; i < tokenCount; i++) {
        predictions.set(i, {
          reuseDistance: 10,
          reuseProbability: 0.5,
          confidence: 0.3,
          method: 'semantic'
        });
      }
      return predictions;
    }
    
    // 找到当前活跃（hot）的region
    const activeRegions = regions.filter(r => r.temperature === 'hot');
    const activeTokenSet = new Set<number>();
    
    for (const region of activeRegions) {
      for (let i = region.startTokenIndex; i < region.endTokenIndex; i++) {
        activeTokenSet.add(i);
      }
    }
    
    // 根据相对位置预测
    for (let tokenId = 0; tokenId < tokenCount; tokenId++) {
      // 找到token所属的region
      let tokenRegion: SemanticRegion | null = null;
      for (const region of regions) {
        if (tokenId >= region.startTokenIndex && tokenId < region.endTokenIndex) {
          tokenRegion = region;
          break;
        }
      }
      
      let reuseDistance: number;
      let reuseProbability: number;
      let confidence: number;
      
      if (tokenRegion) {
        // 有region信息
        const relativePos = tokenId / tokenCount;
        
        // 基于region类型和相对位置
        switch (tokenRegion.type) {
          case 'system_prompt':
            reuseDistance = 1;
            reuseProbability = 0.99;
            confidence = 0.95;
            break;
          case 'active_generation':
            reuseDistance = 2;
            reuseProbability = 0.9;
            confidence = 0.8;
            break;
          case 'reasoning_chain':
          case 'code_context':
          case 'retrieval_chunk':
          case 'dialogue_history':
            if (activeTokenSet.has(tokenId)) {
              reuseDistance = 2;
              reuseProbability = 0.85;
              confidence = 0.7;
            } else {
              reuseDistance = 10;
              reuseProbability = 0.4;
              confidence = 0.5;
            }
            break;
          default:
            reuseDistance = 5;
            reuseProbability = 0.5;
            confidence = 0.4;
        }
      } else {
        // 无region信息，使用位置推断
        reuseDistance = tokenCount - tokenId < 100 ? 2 : 10;
        reuseProbability = tokenCount - tokenId < 100 ? 0.8 : 0.3;
        confidence = 0.3;
      }
      
      predictions.set(tokenId, {
        reuseDistance,
        reuseProbability: Math.round(reuseProbability * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
        method: 'semantic'
      });
    }
    
    return predictions;
  }
  
  /**
   * 方法3：任务模式预测
   * 
   * 根据任务类型和生成阶段预测reuse
   * 
   * math: 推理步骤有周期性回溯
   *   - system prompt: reuseDistance ≈ 1 (每步访问)
   *   - 最近推理步骤: reuseDistance ≈ 2-3 (局部性)
   *   - 早期推理步骤: reuseDistance ≈ 15-20 (可能回溯)
   * 
   * code: 函数调用有模式
   *   - 当前函数签名: reuseDistance ≈ 1
   *   - 当前函数体: reuseDistance ≈ 2
   *   - 其他函数: reuseDistance ≈ 10-50
   *   - import/注释: reuseDistance ≈ ∞
   * 
   * qa: 文档chunk有阶段性
   *   - system prompt: reuseDistance ≈ 1
   *   - 当前相关chunk: reuseDistance ≈ 1-3
   *   - 其他chunk: reuseDistance ≈ ∞
   */
  private taskPatternPredict(
    taskType: TaskType,
    tokenCount: number,
    currentStep: number,
    progress: number
  ): Map<number, TokenReusePrediction> {
    const predictions = new Map<number, TokenReusePrediction>();
    const pattern = TASK_REUSE_PATTERNS[taskType] || TASK_REUSE_PATTERNS.qa;
    
    for (let tokenId = 0; tokenId < tokenCount; tokenId++) {
      const relativePos = tokenId / tokenCount;
      
      let reuseDistance: number;
      let reuseProbability: number;
      let confidence: number;
      
      // 根据任务类型和位置分配模式
      switch (taskType) {
        case 'math':
          if (relativePos < 0.1) {
            // system prompt
            reuseDistance = pattern.systemPrompt.distance;
            reuseProbability = pattern.systemPrompt.probability;
          } else if (relativePos > 0.8) {
            // 最近推理步骤
            reuseDistance = pattern.recentReasoning.distance;
            reuseProbability = pattern.recentReasoning.probability;
          } else if (relativePos > 0.5) {
            // 中间推理
            reuseDistance = pattern.middleReasoning.distance;
            reuseProbability = pattern.middleReasoning.probability;
          } else {
            // 早期推理
            reuseDistance = pattern.earlyReasoning.distance;
            reuseProbability = pattern.earlyReasoning.probability;
          }
          confidence = 0.8;
          break;
          
        case 'code':
          if (relativePos < 0.05) {
            // imports
            reuseDistance = pattern.imports.distance;
            reuseProbability = pattern.imports.probability;
          } else if (relativePos > 0.9) {
            // 当前函数
            reuseDistance = pattern.currentFunction.distance;
            reuseProbability = pattern.currentFunction.probability;
          } else if (relativePos > 0.7) {
            // 当前函数体
            reuseDistance = pattern.currentFunctionBody.distance;
            reuseProbability = pattern.currentFunctionBody.probability;
          } else {
            // 其他函数
            reuseDistance = pattern.otherFunction.distance;
            reuseProbability = pattern.otherFunction.probability;
          }
          confidence = 0.75;
          break;
          
        case 'qa':
          if (relativePos < 0.1) {
            // system prompt
            reuseDistance = pattern.systemPrompt.distance;
            reuseProbability = pattern.systemPrompt.probability;
          } else if (relativePos > 0.8) {
            // 当前chunk
            reuseDistance = pattern.currentChunk.distance;
            reuseProbability = pattern.currentChunk.probability;
          } else if (relativePos > 0.5) {
            // 相关chunk
            reuseDistance = pattern.relatedChunk.distance;
            reuseProbability = pattern.relatedChunk.probability;
          } else {
            // 其他chunk
            reuseDistance = pattern.otherChunk.distance;
            reuseProbability = pattern.otherChunk.probability;
          }
          confidence = 0.7;
          break;
          
        case 'conversation':
          if (relativePos < 0.1) {
            // system prompt
            reuseDistance = pattern.systemPrompt.distance;
            reuseProbability = pattern.systemPrompt.probability;
          } else if (relativePos > 0.8) {
            // 最近轮
            reuseDistance = pattern.recentTurn.distance;
            reuseProbability = pattern.recentTurn.probability;
          } else if (relativePos > 0.5) {
            // 中间轮
            reuseDistance = pattern.middleTurn.distance;
            reuseProbability = pattern.middleTurn.probability;
          } else {
            // 早期轮
            reuseDistance = pattern.earlyTurn.distance;
            reuseProbability = pattern.earlyTurn.probability;
          }
          confidence = 0.75;
          break;
          
        default:
          reuseDistance = 5;
          reuseProbability = 0.5;
          confidence = 0.3;
      }
      
      // 根据生成进度调整（后期更少回溯）
      if (progress > 0.8) {
        reuseProbability *= 0.7; // 后期概率降低
        reuseDistance *= 1.5;    // 距离增加
      }
      
      predictions.set(tokenId, {
        reuseDistance: Math.round(reuseDistance * 10) / 10,
        reuseProbability: Math.round(reuseProbability * 100) / 100,
        confidence: Math.round(confidence * 100) / 100,
        method: 'task-pattern'
      });
    }
    
    return predictions;
  }
  
  /**
   * 综合预测
   * 
   * 策略：加权平均，权重基于置信度
   */
  private combinePredictions(
    statistical: Map<number, TokenReusePrediction>,
    semantic: Map<number, TokenReusePrediction>,
    taskPattern: Map<number, TokenReusePrediction>
  ): Map<number, TokenReusePrediction> {
    const combined = new Map<number, TokenReusePrediction>();
    
    const allTokens = new Set([
      ...Array.from(statistical.keys()),
      ...Array.from(semantic.keys()),
      ...Array.from(taskPattern.keys())
    ]);
    
    for (const tokenId of allTokens) {
      const stats = statistical.get(tokenId);
      const sem = semantic.get(tokenId);
      const task = taskPattern.get(tokenId);
      
      if (!stats && !sem && !task) {
        // 没有任何预测
        combined.set(tokenId, {
          reuseDistance: Infinity,
          reuseProbability: 0,
          confidence: 0,
          method: 'combined'
        });
        continue;
      }
      
      // 加权平均
      let totalWeight = 0;
      let weightedDistance = 0;
      let weightedProbability = 0;
      let maxConfidence = 0;
      const methods: string[] = [];
      
      if (stats) {
        const weight = stats.confidence;
        weightedDistance += stats.reuseDistance * weight;
        weightedProbability += stats.reuseProbability * weight;
        totalWeight += weight;
        maxConfidence = Math.max(maxConfidence, stats.confidence);
        methods.push('statistical');
      }
      
      if (sem) {
        const weight = sem.confidence * 0.8; // 语义权重略低
        weightedDistance += sem.reuseDistance * weight;
        weightedProbability += sem.reuseProbability * weight;
        totalWeight += weight;
        maxConfidence = Math.max(maxConfidence, sem.confidence);
        methods.push('semantic');
      }
      
      if (task) {
        const weight = task.confidence * 0.6; // 任务模式权重最低
        weightedDistance += task.reuseDistance * weight;
        weightedProbability += task.reuseProbability * weight;
        totalWeight += weight;
        maxConfidence = Math.max(maxConfidence, task.confidence);
        methods.push('task-pattern');
      }
      
      if (totalWeight > 0) {
        combined.set(tokenId, {
          reuseDistance: weightedDistance / totalWeight,
          reuseProbability: weightedProbability / totalWeight,
          confidence: maxConfidence,
          method: 'combined'
        });
      }
    }
    
    return combined;
  }
  
  /**
   * 层级别预测
   */
  private predictLayerReuse(
    tokenPredictions: Map<number, TokenReusePrediction>,
    numLayers: number
  ): Map<number, LayerReusePrediction> {
    const layerPredictions = new Map<number, LayerReusePrediction>();
    
    // 简化：假设所有层有相同的token分布
    // 实际中需要根据attention头的分布来计算
    for (let layer = 0; layer < numLayers; layer++) {
      const hotTokens: number[] = [];
      const coldTokens: number[] = [];
      let totalDistance = 0;
      let count = 0;
      
      for (const [tokenId, pred] of tokenPredictions) {
        if (pred.reuseProbability > 0.7) {
          hotTokens.push(tokenId);
        } else if (pred.reuseProbability < 0.3) {
          coldTokens.push(tokenId);
        }
        
        if (pred.reuseDistance !== Infinity) {
          totalDistance += pred.reuseDistance;
          count++;
        }
      }
      
      layerPredictions.set(layer, {
        avgReuseDistance: count > 0 ? totalDistance / count : Infinity,
        hotTokens,
        coldTokens
      });
    }
    
    return layerPredictions;
  }
  
  /**
   * 识别可驱逐的token
   */
  private identifyEvictableTokens(
    predictions: Map<number, TokenReusePrediction>,
    regions: SemanticRegion[]
  ): number[] {
    const evictable: { tokenId: number; score: number }[] = [];
    
    // 找出低reuse概率的token
    for (const [tokenId, pred] of predictions) {
      if (pred.reuseProbability < 0.3 && pred.confidence > 0.5) {
        // 低概率 + 高置信度 = 可驱逐
        evictable.push({ tokenId, score: pred.reuseProbability });
      }
    }
    
    // 按score排序（低score优先驱逐）
    evictable.sort((a, b) => a.score - b.score);
    
    return evictable.map(e => e.tokenId);
  }
  
  /**
   * 更新访问历史
   */
  private updateAccessHistory(historicalAttention: Float64Array[], currentStep: number): void {
    for (const attention of historicalAttention) {
      for (let tokenId = 0; tokenId < attention.length; tokenId++) {
        if (attention[tokenId] > 0.1) {
          const history = this.accessHistory.get(tokenId) || [];
          // 避免重复记录同一step
          if (history.length === 0 || history[history.length - 1] !== currentStep) {
            history.push(currentStep);
            this.accessHistory.set(tokenId, history.slice(-20)); // 保留最近20次
          }
        }
      }
    }
  }
  
  /**
   * 计算全局reuse率
   */
  private calculateGlobalReuseRate(predictions: Map<number, TokenReusePrediction>): number {
    let totalProbability = 0;
    let count = 0;
    
    for (const pred of predictions.values()) {
      if (pred.reuseDistance !== Infinity) {
        totalProbability += pred.reuseProbability;
        count++;
      }
    }
    
    return count > 0 ? totalProbability / count : 0;
  }
  
  /**
   * 获取预测缓存
   */
  getPredictionCache(): Map<number, TokenReusePrediction> {
    return this.predictionCache;
  }
  
  /**
   * 重置状态
   */
  reset(): void {
    this.accessHistory.clear();
    this.emaIntervals.clear();
    this.predictionCache.clear();
  }
}
