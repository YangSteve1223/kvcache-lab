/**
 * Semantic State Agent
 * 
 * 负责：识别当前generation的语义状态
 * 输出：SemanticState（写入Global State Store）
 * 
 * 核心能力：
 * - 识别语义区域（reasoning chain / code context / retrieval chunk等）
 * - 跟踪活跃语义区域的变化
 * - 计算工作集大小
 * - 估计生成进度
 */

import { TaskType } from '../core/types.js';

// ============================================
// 类型定义
// ============================================

/**
 * Semantic Agent 输入接口
 */
export interface SemanticAgentInput {
  tokens: string[];                      // 当前token序列
  taskType: TaskType;
  recentAttention: Float64Array;         // 最近N步的attention分布
  decodeStep: number;
  totalSteps: number;
}

/**
 * 语义区域类型
 */
export type SemanticRegionType = 
  | 'system_prompt' 
  | 'reasoning_chain' 
  | 'code_context' 
  | 'retrieval_chunk' 
  | 'dialogue_history' 
  | 'active_generation';

/**
 * 区域温度（热度）
 */
export type RegionTemperature = 'hot' | 'warm' | 'cold';

/**
 * 语义区域接口
 */
export interface SemanticRegion {
  id: string;
  type: SemanticRegionType;
  startTokenIndex: number;
  endTokenIndex: number;
  layerRange: [number, number];          // 活跃层范围
  accessFrequency: number;               // 访问频率
  lastAccessStep: number;                // 上次访问步
  predictedNextAccess: number;           // 预测下次访问步
  temperature: RegionTemperature;
  retentionPriority: number;            // 0-1, 保留优先级
  metadata?: Record<string, any>;        // 额外元数据
}

/**
 * 语义状态输出
 */
export interface SemanticState {
  activeRegions: SemanticRegion[];
  workingSetTokens: number[];            // 当前工作集的token索引
  reasoningFocus: string;                // 当前推理焦点描述
  generationProgress: number;           // 0-1 生成进度
  taskPhase: 'prefill' | 'decode';      // 当前阶段
  globalAttentionEntropy: number;       // 全局attention熵
}

// ============================================
// 常量定义
// ============================================

// 热度阈值（步数）
const HOT_THRESHOLD = 2;        // 最近2步内被访问 → hot
const WARM_THRESHOLD = 10;      // 最近2-10步被访问 → warm
// 超过10步 → cold

// 区域ID生成计数器
let regionIdCounter = 0;

// 推理关键词（中英文）
const REASONING_KEYWORDS = {
  chinese: ['因为', '由于', '所以', '因此', '综上', '设', '令', '则', '可得', '从而', '可见', '证明', '假设', '推论', '结论'],
  english: ['because', 'therefore', 'hence', 'thus', 'so', 'conclude', 'conclusion', 'assume', 'suppose', 'derive', 'prove', 'henceforth', 'accordingly', 'accordingly', 'consequently']
};

// 代码关键词
const CODE_KEYWORDS = ['def', 'class', 'function', 'if', 'for', 'while', 'return', 'import', 'from', 'const', 'let', 'var', 'async', 'await', 'try', 'catch'];

// ============================================
// SemanticAgent 类实现
// ============================================

export class SemanticAgent {
  private regions: SemanticRegion[] = [];
  private regionHistory: Map<string, number[]> = new Map(); // 记录每个region的访问历史
  private lastAttentionEntropy: number = 0;
  
  /**
   * 核心方法：分析语义状态
   * 
   * @param input 输入参数
   * @returns 语义状态（写入Global State Store）
   */
  analyze(input: SemanticAgentInput): SemanticState {
    const { tokens, taskType, recentAttention, decodeStep, totalSteps } = input;
    
    // 1. 识别语义区域
    this.regions = this.identifyRegions(tokens, taskType);
    
    // 2. 更新活跃度
    this.updateActivity(this.regions, decodeStep, recentAttention);
    
    // 3. 计算工作集
    const workingSetTokens = this.computeWorkingSet(this.regions, 512); // 默认工作集大小512 tokens
    
    // 4. 估计生成进度
    const generationProgress = this.estimateProgress(decodeStep, totalSteps, this.lastAttentionEntropy);
    
    // 5. 确定推理焦点
    const reasoningFocus = this.determineReasoningFocus(this.regions, taskType);
    
    // 6. 计算全局attention熵
    this.lastAttentionEntropy = this.calculateAttentionEntropy(recentAttention);
    
    return {
      activeRegions: this.regions,
      workingSetTokens,
      reasoningFocus,
      generationProgress,
      taskPhase: decodeStep === 0 ? 'prefill' : 'decode',
      globalAttentionEntropy: this.lastAttentionEntropy
    };
  }
  
  /**
   * 区域识别
   * 
   * 根据任务类型识别不同的语义区域
   */
  private identifyRegions(tokens: string[], taskType: TaskType): SemanticRegion[] {
    switch (taskType) {
      case 'math':
        return this.identifyMathRegions(tokens);
      case 'code':
        return this.identifyCodeRegions(tokens);
      case 'qa':
        return this.identifyQARegions(tokens);
      case 'conversation':
        return this.identifyConversationRegions(tokens);
      default:
        return this.identifyGenericRegions(tokens);
    }
  }
  
  /**
   * 数学/推理任务区域识别
   * 
   * 策略：
   * - system prompt: token 0 - 第一个推理关键词之前
   * - reasoning chain: 每个推理步骤一个region，检测关键词形成边界
   */
  private identifyMathRegions(tokens: string[]): SemanticRegion[] {
    const regions: SemanticRegion[] = [];
    
    // 检测推理关键词位置
    const reasoningBoundaries: number[] = [0];
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i].toLowerCase();
      for (const keyword of [...REASONING_KEYWORDS.chinese, ...REASONING_KEYWORDS.english]) {
        if (token.includes(keyword.toLowerCase())) {
          reasoningBoundaries.push(i);
          break;
        }
      }
    }
    reasoningBoundaries.push(tokens.length);
    
    // 去重并排序
    const uniqueBoundaries = [...new Set(reasoningBoundaries)].sort((a, b) => a - b);
    
    // 创建reasoning_chain regions
    for (let i = 0; i < uniqueBoundaries.length - 1; i++) {
      const start = uniqueBoundaries[i];
      const end = uniqueBoundaries[i + 1];
      
      if (end - start < 5) continue; // 忽略太短的区域
      
      regions.push({
        id: `math-reasoning-${i}`,
        type: 'reasoning_chain',
        startTokenIndex: start,
        endTokenIndex: end,
        layerRange: [0, 31], // 全层活跃
        accessFrequency: 0,
        lastAccessStep: -1,
        predictedNextAccess: -1,
        temperature: 'cold',
        retentionPriority: this.calculateMathRetentionPriority(start, tokens.length)
      });
    }
    
    return regions.length > 0 ? regions : this.identifyGenericRegions(tokens);
  }
  
  /**
   * 代码任务区域识别
   * 
   * 策略：
   * - import区域: 代码头部import语句
   * - 函数区域: 每个def/class/function形成独立region
   * - 注释区域: 标记但不分配高优先级
   */
  private identifyCodeRegions(tokens: string[]): SemanticRegion[] {
    const regions: SemanticRegion[] = [];
    const codeBlocks: { start: number; end: number; type: string }[] = [];
    
    let currentBlockStart = -1;
    let currentBlockType = '';
    let indentStack: number[] = [0];
    
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i].toLowerCase().trim();
      
      // 检测代码关键词
      if (token === 'def' || token === 'class' || token === 'function') {
        if (currentBlockStart >= 0) {
          codeBlocks.push({ start: currentBlockStart, end: i, type: currentBlockType });
        }
        currentBlockStart = i;
        currentBlockType = token === 'class' ? 'class' : 'function';
      } else if (token === 'import' || token === 'from') {
        if (currentBlockStart >= 0 && currentBlockType === 'imports') {
          // 继续import区域
        } else {
          if (currentBlockStart >= 0 && currentBlockType === 'imports') {
            codeBlocks.push({ start: currentBlockStart, end: i, type: currentBlockType });
          }
          currentBlockStart = i;
          currentBlockType = 'imports';
        }
      } else if (['if', 'for', 'while', 'try'].includes(token)) {
        indentStack.push(indentStack[indentStack.length - 1] + 1);
      } else if (token === 'return' || token === '}') {
        if (indentStack.length > 1) indentStack.pop();
      }
    }
    
    // 添加最后一个块
    if (currentBlockStart >= 0) {
      codeBlocks.push({ start: currentBlockStart, end: tokens.length, type: currentBlockType });
    }
    
    // 转换为regions
    for (const block of codeBlocks) {
      regions.push({
        id: `code-${block.type}-${block.start}`,
        type: 'code_context',
        startTokenIndex: block.start,
        endTokenIndex: block.end,
        layerRange: [0, 31],
        accessFrequency: 0,
        lastAccessStep: -1,
        predictedNextAccess: -1,
        temperature: 'cold',
        retentionPriority: this.calculateCodeRetentionPriority(block.type, block.start, tokens.length)
      });
    }
    
    return regions.length > 0 ? regions : this.identifyGenericRegions(tokens);
  }
  
  /**
   * QA任务区域识别
   * 
   * 策略：
   * - system prompt: 始终hot
   * - retrieval chunks: 根据位置分配优先级
   * - 当前相关chunk: hot
   */
  private identifyQARegions(tokens: string[]): SemanticRegion[] {
    const regions: SemanticRegion[] = [];
    
    // 假设tokens序列包含分段标记
    // 简单的等分策略（实际中应该检测分段标记）
    const chunkSize = Math.max(50, Math.floor(tokens.length / 4));
    
    for (let i = 0; i < tokens.length; i += chunkSize) {
      const start = i;
      const end = Math.min(i + chunkSize, tokens.length);
      
      // system prompt区域（token 0-100左右）
      let type: SemanticRegionType = 'retrieval_chunk';
      let priority = 0.5;
      
      if (start < 100) {
        type = 'system_prompt';
        priority = 1.0;
      } else if (start >= tokens.length - chunkSize * 2) {
        // 最近2个chunk更可能相关
        priority = 0.7;
      }
      
      regions.push({
        id: `qa-chunk-${i}`,
        type,
        startTokenIndex: start,
        endTokenIndex: end,
        layerRange: [0, 31],
        accessFrequency: 0,
        lastAccessStep: -1,
        predictedNextAccess: -1,
        temperature: type === 'system_prompt' ? 'hot' : 'cold',
        retentionPriority: priority
      });
    }
    
    return regions;
  }
  
  /**
   * 对话任务区域识别
   * 
   * 策略：每轮对话一个region
   */
  private identifyConversationRegions(tokens: string[]): SemanticRegion[] {
    const regions: SemanticRegion[] = [];
    
    // 检测对话标记（简化：查找"User:" / "Assistant:" 等）
    const turnBoundaries: number[] = [0];
    const turnPatterns = ['user', 'human', 'assistant', 'bot', 'system'];
    
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i].toLowerCase();
      for (const pattern of turnPatterns) {
        if (token.includes(pattern + ':') || token.includes(pattern + '：')) {
          turnBoundaries.push(i);
          break;
        }
      }
    }
    turnBoundaries.push(tokens.length);
    
    // 去重
    const uniqueBoundaries = [...new Set(turnBoundaries)].sort((a, b) => a - b);
    
    for (let i = 0; i < uniqueBoundaries.length - 1; i++) {
      const start = uniqueBoundaries[i];
      const end = uniqueBoundaries[i + 1];
      
      regions.push({
        id: `conv-turn-${i}`,
        type: 'dialogue_history',
        startTokenIndex: start,
        endTokenIndex: end,
        layerRange: [0, 31],
        accessFrequency: 0,
        lastAccessStep: -1,
        predictedNextAccess: -1,
        temperature: i === uniqueBoundaries.length - 2 ? 'hot' : 'cold',
        retentionPriority: i === uniqueBoundaries.length - 2 ? 0.9 : 0.4
      });
    }
    
    return regions.length > 0 ? regions : this.identifyGenericRegions(tokens);
  }
  
  /**
   * 通用区域识别（fallback）
   */
  private identifyGenericRegions(tokens: string[]): SemanticRegion[] {
    // 简单等分
    const chunkSize = 100;
    const regions: SemanticRegion[] = [];
    
    for (let i = 0; i < tokens.length; i += chunkSize) {
      regions.push({
        id: `generic-${i}`,
        type: 'active_generation',
        startTokenIndex: i,
        endTokenIndex: Math.min(i + chunkSize, tokens.length),
        layerRange: [0, 31],
        accessFrequency: 0,
        lastAccessStep: -1,
        predictedNextAccess: -1,
        temperature: 'cold',
        retentionPriority: 0.5
      });
    }
    
    return regions;
  }
  
  /**
   * 工作集计算
   * 
   * 根据区域优先级和预算，计算当前应保留的token
   */
  private computeWorkingSet(regions: SemanticRegion[], budget: number): number[] {
    // 按优先级排序
    const sortedRegions = [...regions].sort((a, b) => b.retentionPriority - a.retentionPriority);
    
    const workingSet: number[] = [];
    let remainingBudget = budget;
    
    for (const region of sortedRegions) {
      if (remainingBudget <= 0) break;
      
      const regionSize = region.endTokenIndex - region.startTokenIndex;
      const tokensToTake = Math.min(regionSize, Math.ceil(remainingBudget * region.retentionPriority));
      
      for (let i = region.startTokenIndex; i < region.startTokenIndex + tokensToTake; i++) {
        if (workingSet.length >= budget) break;
        workingSet.push(i);
      }
      
      remainingBudget -= tokensToTake;
    }
    
    return workingSet.sort((a, b) => a - b);
  }
  
  /**
   * 活跃度更新
   * 
   * 根据attention分布更新区域的热度
   */
  private updateActivity(
    regions: SemanticRegion[], 
    currentStep: number, 
    attention: Float64Array
  ): void {
    // 统计attention分布
    const regionAccessCounts = new Map<string, number>();
    
    for (let i = 0; i < attention.length && i < regions.length; i++) {
      const region = regions[i];
      const accessWeight = attention[i];
      
      const currentCount = regionAccessCounts.get(region.id) || 0;
      regionAccessCounts.set(region.id, currentCount + accessWeight);
      
      // 更新访问历史
      if (accessWeight > 0.1) { // 阈值：attention > 0.1 才算有效访问
        const history = this.regionHistory.get(region.id) || [];
        history.push(currentStep);
        this.regionHistory.set(region.id, history.slice(-20)); // 保留最近20次访问
        
        region.lastAccessStep = currentStep;
        region.accessFrequency++;
      }
    }
    
    // 更新温度
    for (const region of regions) {
      region.temperature = this.calculateTemperature(region.lastAccessStep, currentStep);
      
      // 预测下次访问
      const history = this.regionHistory.get(region.id);
      if (history && history.length >= 2) {
        const intervals: number[] = [];
        for (let i = 1; i < history.length; i++) {
          intervals.push(history[i] - history[i - 1]);
        }
        // EMA预测
        const avgInterval = intervals.reduce((a, b) => a + b, 0) / intervals.length;
        region.predictedNextAccess = region.lastAccessStep + Math.round(avgInterval);
      }
    }
  }
  
  /**
   * 计算区域温度
   */
  private calculateTemperature(lastAccessStep: number, currentStep: number): RegionTemperature {
    if (lastAccessStep < 0) return 'cold';
    
    const stepsSinceAccess = currentStep - lastAccessStep;
    
    if (stepsSinceAccess <= HOT_THRESHOLD) return 'hot';
    if (stepsSinceAccess <= WARM_THRESHOLD) return 'warm';
    return 'cold';
  }
  
  /**
   * 生成进度估计
   */
  private estimateProgress(
    decodeStep: number, 
    totalSteps: number, 
    attentionEntropy: number
  ): number {
    if (totalSteps === 0) return 0;
    
    // 基础进度
    let progress = decodeStep / totalSteps;
    
    // 根据attention熵调整
    // 低熵（高聚焦）→ 早期阶段概率高
    // 高熵（分散）→ 可能是中间阶段
    if (attentionEntropy < 2) {
      // 低熵：可能是早期（刚进入decode），进度可能比实际略低
      progress = progress * 0.9;
    } else if (attentionEntropy > 4) {
      // 高熵：可能是后期（需要检索），进度可能比实际略高
      progress = Math.min(1, progress * 1.1);
    }
    
    return Math.max(0, Math.min(1, progress));
  }
  
  /**
   * 确定推理焦点
   */
  private determineReasoningFocus(regions: SemanticRegion[], taskType: TaskType): string {
    // 找到当前最热的region
    const hotRegions = regions.filter(r => r.temperature === 'hot');
    
    if (hotRegions.length > 0) {
      const focusedRegion = hotRegions.sort((a, b) => 
        b.accessFrequency - a.accessFrequency
      )[0];
      
      return `${focusedRegion.type}: tokens ${focusedRegion.startTokenIndex}-${focusedRegion.endTokenIndex}`;
    }
    
    return `${taskType}: no active focus`;
  }
  
  /**
   * 计算attention熵
   */
  private calculateAttentionEntropy(attention: Float64Array): number {
    if (attention.length === 0) return 0;
    
    let entropy = 0;
    const sum = attention.reduce((a, b) => a + b, 0);
    
    if (sum === 0) return 0;
    
    for (const p of attention) {
      const normalized = p / sum;
      if (normalized > 0) {
        entropy -= normalized * Math.log2(normalized);
      }
    }
    
    return entropy;
  }
  
  /**
   * 数学任务保留优先级计算
   */
  private calculateMathRetentionPriority(tokenPosition: number, totalTokens: number): number {
    const relativePosition = tokenPosition / totalTokens;
    
    // system prompt 和 最近推理步骤 高优先级
    if (relativePosition < 0.1) return 1.0; // system prompt
    if (relativePosition > 0.8) return 0.9; // 最近推理步骤
    if (relativePosition > 0.5) return 0.7; // 中间步骤
    return 0.4; // 早期步骤
  }
  
  /**
   * 代码任务保留优先级计算
   */
  private calculateCodeRetentionPriority(
    codeType: string, 
    tokenPosition: number, 
    totalTokens: number
  ): number {
    const relativePosition = tokenPosition / totalTokens;
    
    switch (codeType) {
      case 'imports': return 0.3; // imports低优先级
      case 'class': return 0.8;   // class定义高优先级
      case 'function': return 0.9; // 当前函数最高
      default: return 0.5;
    }
  }
  
  /**
   * 获取当前区域列表
   */
  getRegions(): SemanticRegion[] {
    return this.regions;
  }
  
  /**
   * 重置状态
   */
  reset(): void {
    this.regions = [];
    this.regionHistory.clear();
    this.lastAttentionEntropy = 0;
  }
}
