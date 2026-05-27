/**
 * 任务分类器 - 增强版
 * 支持离线规则模式和在线API模式
 * 
 * 特点：
 * - 大幅扩充关键词库
 * - 加权评分机制
 * - 结构化特征检测
 * - 上下文增强
 */

import {
  TaskType,
  ClassificationResult,
  TaskClassifierOptions
} from '../core/types.js';

import OpenAI from 'openai';

// ============================================
// 关键词库 - 加权分类
// ============================================

// 数学类关键词 (权重: 高=3, 中=2, 低=1)
interface KeywordEntry {
  keyword: string;
  weight: number;
}

const MATH_KEYWORDS: KeywordEntry[] = [
  // 高权重关键词
  { keyword: '积分', weight: 3 },
  { keyword: '微分', weight: 3 },
  { keyword: '求导', weight: 3 },
  { keyword: '方程', weight: 3 },
  { keyword: '求解', weight: 3 },
  { keyword: '求积分', weight: 3 },
  { keyword: '矩阵', weight: 3 },
  { keyword: '特征值', weight: 3 },
  { keyword: '特征向量', weight: 3 },
  { keyword: '行列式', weight: 3 },
  { keyword: '证明', weight: 3 },
  { keyword: '求证', weight: 3 },
  { keyword: '定理', weight: 3 },
  { keyword: '级数', weight: 3 },
  { keyword: '泰勒', weight: 3 },
  { keyword: '傅里叶', weight: 3 },
  { keyword: '拉普拉斯', weight: 3 },
  { keyword: '梯度', weight: 3 },
  { keyword: '偏导', weight: 3 },
  { keyword: '极值', weight: 3 },
  { keyword: '凸函数', weight: 3 },
  { keyword: '∑', weight: 3 },
  { keyword: '∫', weight: 3 },
  { keyword: '∂', weight: 3 },
  { keyword: '√', weight: 3 },
  // 中权重关键词
  { keyword: '计算', weight: 2 },
  { keyword: '推导', weight: 2 },
  { keyword: '公式', weight: 2 },
  { keyword: '概率', weight: 2 },
  { keyword: '统计', weight: 2 },
  { keyword: '向量', weight: 2 },
  { keyword: '函数', weight: 2 },
  { keyword: '极限', weight: 2 },
  { keyword: '优化', weight: 2 },
  { keyword: '算法复杂度', weight: 2 },
  { keyword: '时间复杂度', weight: 2 },
  { keyword: '空间复杂度', weight: 2 },
  { keyword: 'O(n', weight: 2 },
  { keyword: 'log', weight: 2 },
  // 低权重关键词
  { keyword: '数学', weight: 1 },
  { keyword: '代数', weight: 1 },
  { keyword: '几何', weight: 1 },
  { keyword: '三角', weight: 1 },
  { keyword: 'calculus', weight: 2 },
  { keyword: 'derivative', weight: 3 },
  { keyword: 'integral', weight: 3 },
  { keyword: 'matrix', weight: 2 },
  { keyword: 'theorem', weight: 2 },
  { keyword: 'proof', weight: 2 },
  { keyword: 'equation', weight: 2 },
  { keyword: 'probability', weight: 2 },
  { keyword: 'optimize', weight: 1 },
  { keyword: 'gradient', weight: 2 },
  { keyword: 'linear algebra', weight: 2 },
];

// 代码类关键词 (权重: 高=3, 中=2, 低=1)
const CODE_KEYWORDS: KeywordEntry[] = [
  // 高权重关键词 - 编程语言关键字和语法
  { keyword: 'def ', weight: 3 },
  { keyword: 'class ', weight: 3 },
  { keyword: 'function ', weight: 3 },
  { keyword: 'async ', weight: 3 },
  { keyword: 'await ', weight: 3 },
  { keyword: 'import ', weight: 3 },
  { keyword: 'export ', weight: 3 },
  { keyword: 'return ', weight: 3 },
  { keyword: 'const ', weight: 3 },
  { keyword: 'let ', weight: 3 },
  { keyword: 'var ', weight: 3 },
  { keyword: 'try {', weight: 3 },
  { keyword: 'catch ', weight: 3 },
  { keyword: 'throw ', weight: 3 },
  { keyword: 'interface ', weight: 3 },
  { keyword: 'enum ', weight: 3 },
  { keyword: 'lambda ', weight: 3 },
  { keyword: 'yield ', weight: 3 },
  { keyword: 'self.', weight: 3 },
  { keyword: 'this.', weight: 3 },
  { keyword: 'super.', weight: 3 },
  { keyword: 'new ', weight: 3 },
  { keyword: 'static ', weight: 3 },
  { keyword: 'abstract ', weight: 3 },
  { keyword: 'public ', weight: 2 },
  { keyword: 'private ', weight: 2 },
  { keyword: 'protected ', weight: 2 },
  { keyword: 'console.log', weight: 3 },
  { keyword: 'print(', weight: 3 },
  { keyword: 'print ', weight: 2 },
  { keyword: '=>', weight: 3 },
  { keyword: '->', weight: 3 },
  { keyword: '::', weight: 3 },
  { keyword: '===', weight: 3 },
  { keyword: '!==', weight: 3 },
  { keyword: 'git ', weight: 2 },
  { keyword: 'docker ', weight: 2 },
  { keyword: 'npm ', weight: 2 },
  { keyword: 'pip ', weight: 2 },
  { keyword: 'yarn ', weight: 2 },
  // 高权重 - 完整短语
  { keyword: '写一个函数', weight: 3 },
  { keyword: '实现一个', weight: 3 },
  { keyword: '写一段代码', weight: 3 },
  { keyword: '写一个class', weight: 3 },
  { keyword: '修复这个bug', weight: 3 },
  { keyword: 'debug', weight: 3 },
  { keyword: 'API调用', weight: 2 },
  { keyword: 'API', weight: 2 },
  // 中权重关键词
  { keyword: '函数', weight: 2 },
  { keyword: '代码', weight: 2 },
  { keyword: '实现', weight: 2 },
  { keyword: '编程', weight: 2 },
  { keyword: '算法', weight: 2 },
  { keyword: '递归', weight: 2 },
  { keyword: '循环', weight: 2 },
  { keyword: '编译', weight: 2 },
  { keyword: '运行', weight: 2 },
  { keyword: '部署', weight: 2 },
  { keyword: '框架', weight: 2 },
  { keyword: '库', weight: 2 },
  { keyword: '模块', weight: 2 },
  { keyword: '接口', weight: 2 },
  { keyword: '继承', weight: 2 },
  { keyword: '多态', weight: 2 },
  { keyword: '异步', weight: 2 },
  { keyword: '回调', weight: 2 },
  { keyword: '闭包', weight: 2 },
  { keyword: '装饰器', weight: 2 },
  { keyword: '泛型', weight: 2 },
  { keyword: 'promise', weight: 3 },
  { keyword: 'method', weight: 2 },
  { keyword: 'variable', weight: 2 },
  { keyword: 'loop', weight: 2 },
  { keyword: 'recursion', weight: 2 },
  { keyword: 'compile', weight: 2 },
  { keyword: 'runtime', weight: 2 },
  { keyword: 'deploy', weight: 2 },
  { keyword: 'framework', weight: 2 },
  { keyword: 'library', weight: 2 },
  { keyword: 'module', weight: 2 },
  { keyword: 'endpoint', weight: 2 },
  { keyword: 'request', weight: 2 },
  { keyword: 'response', weight: 2 },
  { keyword: 'database', weight: 2 },
  { keyword: 'cache', weight: 2 },
  // 低权重关键词
  { keyword: '程序', weight: 1 },
  { keyword: 'python', weight: 2 },
  { keyword: 'javascript', weight: 2 },
  { keyword: 'typescript', weight: 2 },
  { keyword: 'java', weight: 1 },
  { keyword: 'rust', weight: 1 },
  { keyword: 'go', weight: 1 },
  { keyword: 'sql', weight: 1 },
  { keyword: 'html', weight: 1 },
  { keyword: 'css', weight: 1 },
  { keyword: 'server', weight: 1 },
  { keyword: 'client', weight: 1 },
];

// QA类关键词 (权重: 高=3, 中=2, 低=1)
const QA_KEYWORDS: KeywordEntry[] = [
  // 高权重关键词
  { keyword: '什么是', weight: 3 },
  { keyword: '解释一下', weight: 3 },
  { keyword: '有什么区别', weight: 3 },
  { keyword: 'why', weight: 3 },
  { keyword: 'how to', weight: 3 },
  { keyword: 'explain', weight: 3 },
  { keyword: 'difference between', weight: 3 },
  { keyword: '总结一下', weight: 3 },
  { keyword: '请说明', weight: 3 },
  { keyword: '介绍一下', weight: 3 },
  { keyword: 'define', weight: 3 },
  { keyword: 'definition', weight: 3 },
  { keyword: '概念', weight: 3 },
  // 中权重关键词
  { keyword: '为什么', weight: 2 },
  { keyword: '如何', weight: 2 },
  { keyword: '怎么', weight: 2 },
  { keyword: '怎样', weight: 2 },
  { keyword: '总结', weight: 2 },
  { keyword: '解释', weight: 2 },
  { keyword: '区别', weight: 2 },
  { keyword: '比较', weight: 2 },
  { keyword: '优缺点', weight: 2 },
  { keyword: '优点', weight: 2 },
  { keyword: '缺点', weight: 2 },
  { keyword: '意义', weight: 2 },
  { keyword: '影响', weight: 2 },
  { keyword: '分析', weight: 2 },
  { keyword: '评价', weight: 2 },
  { keyword: '建议', weight: 2 },
  { keyword: '理解', weight: 2 },
  { keyword: '定义', weight: 2 },
  { keyword: '原理', weight: 2 },
  { keyword: '机制', weight: 2 },
  { keyword: '流程', weight: 2 },
  { keyword: '步骤', weight: 2 },
  { keyword: '告诉我', weight: 2 },
  { keyword: '请介绍', weight: 2 },
  { keyword: '介绍一下', weight: 2 },
  { keyword: 'what is', weight: 3 },
  { keyword: 'summarize', weight: 2 },
  { keyword: 'meaning', weight: 2 },
  { keyword: 'impact', weight: 2 },
  { keyword: 'analyze', weight: 2 },
  { keyword: 'evaluate', weight: 2 },
  { keyword: 'suggest', weight: 2 },
  { keyword: 'understand', weight: 2 },
  { keyword: 'concept', weight: 2 },
  { keyword: 'principle', weight: 2 },
  { keyword: 'mechanism', weight: 2 },
  { keyword: 'process', weight: 2 },
  { keyword: 'tell me about', weight: 3 },
  // 低权重关键词
  { keyword: '特点', weight: 1 },
  { keyword: '作用', weight: 1 },
  { keyword: 'describe', weight: 1 },
];

// Conversation类关键词 (权重)
const CONVERSATION_KEYWORDS: KeywordEntry[] = [
  // 高权重
  { keyword: '你好', weight: 3 },
  { keyword: '您好', weight: 3 },
  { keyword: 'hi', weight: 2 },
  { keyword: 'hello', weight: 2 },
  { keyword: 'hey', weight: 2 },
  { keyword: 'thanks', weight: 3 },
  { keyword: 'thank you', weight: 3 },
  { keyword: '谢谢', weight: 3 },
  { keyword: '再见', weight: 3 },
  { keyword: 'bye', weight: 2 },
  { keyword: 'goodbye', weight: 2 },
  { keyword: 'please', weight: 2 },
  { keyword: '请', weight: 1 },
  { keyword: '能帮我', weight: 2 },
  { keyword: '帮我', weight: 1 },
  { keyword: 'can you', weight: 2 },
  { keyword: 'help me', weight: 2 },
  { keyword: 'i want', weight: 2 },
  { keyword: '我想', weight: 1 },
  { keyword: 'what do you think', weight: 3 },
  { keyword: '你怎么看', weight: 3 },
  { keyword: '聊聊', weight: 2 },
  { keyword: "let's talk", weight: 3 },
  { keyword: 'how are you', weight: 3 },
  { keyword: '天气', weight: 2 },
  { keyword: 'good morning', weight: 3 },
  { keyword: 'good night', weight: 3 },
  { keyword: '晚安', weight: 2 },
  { keyword: '早上好', weight: 2 },
  // 低权重
  { keyword: '吗', weight: 1 },
  { keyword: '呢', weight: 1 },
];

// ============================================
// 结构化特征检测
// ============================================

// 代码结构特征
const CODE_PATTERNS = [
  /\{[\s\S]*\}/,                    // 大括号块
  /\[[\s\S]*\]/,                    // 方括号块
  /<\w+[\s\S]*<\/\w+>/,            // HTML/XML标签
  /=\s*[=:]/,                       // 赋值
  /==\s*/,                          // 相等比较
  /===\s*/,                         // 严格相等
  /=>\s*[{(]/,                      // 箭头函数
  /::\s*/,                          // 双冒号
  /->\s*/,                          // 箭头
  /def\s+\w+\s*\(/,                // Python函数定义
  /class\s+\w+\s*[:{]/,            // 类定义
  /function\s+\w+\s*\(/,           // JS函数定义
  /import\s+[\w{}]+from/,          // ES6 import
  /export\s+(default\s+)?/,         // ES6 export
  /async\s+(function|\()/i,        // async函数
  /await\s+/,                       // await
  /try\s*\{/,                       // try块
  /catch\s*\(/,                     // catch块
  /console\.\w+/,                   // console方法
  /print\s*\(/,                     // print语句
  /\w+\.\w+\s*\([^)]*\)\s*;/,       // 方法调用
  /^\s{2,}/m,                       // 缩进行
  /#include\s*</,                   // C/C++头文件
  /public\s+(static\s+)?void/i,    // Java方法
  /func\s+\w+\s*\([^)]*\)/i,       // Go/Rust函数
  /SELECT\s+.+\s+FROM/i,           // SQL查询
  /INSERT\s+INTO/i,                // SQL插入
  /CREATE\s+TABLE/i,               // SQL建表
];

// 数学符号特征
const MATH_PATTERNS = [
  /[∑∫∂∇∏√±≤≥∞∈∉∀∃]/,           // 数学符号
  /\^/,                             // 指数符号
  /_\d/,                            // 下标
  /\^2/,                           // 平方
  /f\s*\(\s*\w+\s*\)/,             // f(x)函数
  /log_\d/,                         // 对数
  /sin|cos|tan|arcsin|arccos|arctan/i, // 三角函数
  /lim\s+/,                         // 极限
  /→\s*\d/,                         // 极限趋向
  /\\frac\{/,                       // LaTeX分数
  /\\sum/,                          // LaTeX求和
  /\\int/,                          // LaTeX积分
  /\|\s*\w+\s*\|/,                 // 绝对值/行列式
  /\w+\s*×\s*\w+/,                 // 乘法
  /\w+\s*÷\s*\w+/,                 // 除法
  /matrix/i,                        // 矩阵
  /vector/i,                        // 向量
  /eigenvalue/i,                    // 特征值
  /derivative/i,                    // 导数
  /integral/i,                      // 积分
];

// QA问句特征
const QA_PATTERNS = [
  /\?$/m,                           // 以问号结尾
  /？$/m,                           // 中文问号结尾
  /\?\s*$/m,                        // 末尾问号
  /^(what|why|how|when|where|who|which|is|are|can|could|would|should)/im, // 英文问句开头
  /^(什么|为什么|如何|怎么|怎样|谁|哪里|哪个|是不是|能不能)/,             // 中文问句开头
  /请说明/,                         // 请求说明
  /请解释/,                         // 请求解释
  /请分析/,                         // 请求分析
  /请比较/,                         // 请求比较
  /请评价/,                         // 请求评价
];

// ============================================
// 上下文增强特征
// ============================================

// 代码块标记
const CODE_BLOCK_PATTERN = /```[\s\S]*?```|`[^`]+`/;
// 数学公式标记
const MATH_BLOCK_PATTERN = /\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)/;
// 列表模式
const LIST_PATTERN = /^\s*[-*•]\s+/m;

// ============================================
// 分类器实现
// ============================================

/**
 * 计算加权关键词得分
 */
function calculateWeightedScore(text: string, keywords: KeywordEntry[]): number {
  let score = 0;
  const lowerText = text.toLowerCase();
  
  for (const { keyword, weight } of keywords) {
    if (lowerText.includes(keyword.toLowerCase())) {
      score += weight;
    }
  }
  
  return score;
}

/**
 * 检测代码结构特征
 */
function detectCodeFeatures(text: string): number {
  let score = 0;
  for (const pattern of CODE_PATTERNS) {
    if (pattern.test(text)) {
      score += 2;
    }
  }
  return score;
}

/**
 * 检测数学特征
 */
function detectMathFeatures(text: string): number {
  let score = 0;
  for (const pattern of MATH_PATTERNS) {
    if (pattern.test(text)) {
      score += 2;
    }
  }
  return score;
}

/**
 * 检测QA特征
 */
function detectQAFeatures(text: string): number {
  let score = 0;
  for (const pattern of QA_PATTERNS) {
    if (pattern.test(text)) {
      score += 2;
    }
  }
  return score;
}

/**
 * 上下文增强检测
 */
function detectContextFeatures(text: string): { math: number; code: number; qa: number; conversation: number } {
  const result = { math: 0, code: 0, qa: 0, conversation: 0 };
  
  // 代码块标记
  if (CODE_BLOCK_PATTERN.test(text)) {
    result.code += 10;
  }
  
  // 数学公式标记
  if (MATH_BLOCK_PATTERN.test(text)) {
    result.math += 10;
  }
  
  // 短文本+问候
  if (text.length < 50) {
    if (/^(你好|hi|hello|hey|good|morning|afternoon|evening)/i.test(text)) {
      result.conversation += 5;
    }
  }
  
  // 问号结尾
  if (/\?$|？$/.test(text.trim())) {
    result.qa += 3;
  }
  
  return result;
}

/**
 * 增强版规则分类
 */
function classifyByRulesEnhanced(input: string): { taskType: TaskType; confidence: number } {
  const scores = {
    math: 0,
    code: 0,
    qa: 0,
    conversation: 0,
  };
  
  // 1. 加权关键词得分
  scores.math += calculateWeightedScore(input, MATH_KEYWORDS);
  scores.code += calculateWeightedScore(input, CODE_KEYWORDS);
  scores.qa += calculateWeightedScore(input, QA_KEYWORDS);
  scores.conversation += calculateWeightedScore(input, CONVERSATION_KEYWORDS);
  
  // 2. 结构化特征检测
  scores.code += detectCodeFeatures(input);
  scores.math += detectMathFeatures(input);
  scores.qa += detectQAFeatures(input);
  
  // 3. 上下文增强
  const context = detectContextFeatures(input);
  scores.math += context.math;
  scores.code += context.code;
  scores.qa += context.qa;
  scores.conversation += context.conversation;
  
  // 4. 找出最高分
  const entries = Object.entries(scores) as [TaskType, number][];
  entries.sort((a, b) => b[1] - a[1]);
  
  const [topType, topScore] = entries[0];
  const secondScore = entries[1][1];
  
  // 如果最高分为0或很低，返回conversation
  if (topScore === 0) {
    return { taskType: 'conversation', confidence: 0.5 };
  }
  
  // 得分差异
  const scoreDiff = topScore - secondScore;
  
  // 计算置信度
  let confidence: number;
  if (scoreDiff >= 8 || topScore >= 15) {
    confidence = 0.95;
  } else if (scoreDiff >= 5 || topScore >= 10) {
    confidence = 0.9;
  } else if (scoreDiff >= 3 || topScore >= 5) {
    confidence = 0.8;
  } else if (scoreDiff >= 1 || topScore >= 2) {
    confidence = 0.7;
  } else {
    confidence = 0.6;
  }
  
  return {
    taskType: topType,
    confidence: Math.min(confidence, 0.98),
  };
}

/**
 * API模式分类 - 调用DeepSeek进行分类
 */
async function classifyByAPI(
  input: string,
  apiKey: string,
  timeout: number
): Promise<ClassificationResult> {
  const startTime = Date.now();
  
  // 截取前200个字符
  const truncatedInput = input.substring(0, 200);
  
  const client = new OpenAI({
    apiKey: apiKey,
    baseURL: 'https://api.deepseek.com',
  });
  
  const prompt = `你是一个LLM请求任务分类器。根据以下用户输入，判断其任务类型。
只返回一个JSON: {"taskType": "math|code|qa|conversation", "confidence": 0.0-1.0}

用户输入：
${truncatedInput}`;
  
  try {
    const response = await client.chat.completions.create({
      model: 'deepseek-chat',
      messages: [
        { role: 'user', content: prompt }
      ],
      temperature: 0.1,
      max_tokens: 100,
    }, {
      timeout: timeout * 1000,
    });
    
    const content = response.choices[0]?.message?.content || '';
    const latencyMs = Date.now() - startTime;
    
    // 解析返回的JSON
    try {
      const jsonMatch = content.match(/\{[\s\S]*\}/);
      if (jsonMatch) {
        const result = JSON.parse(jsonMatch[0]);
        return {
          taskType: result.taskType as TaskType,
          confidence: Math.max(0, Math.min(1, result.confidence)),
          method: 'api',
          latencyMs,
        };
      }
    } catch {
      // JSON解析失败
    }
    
    return {
      taskType: 'unknown',
      confidence: 0,
      method: 'api',
      latencyMs,
    };
  } catch (error) {
    return {
      taskType: 'unknown',
      confidence: 0,
      method: 'api',
      latencyMs: Date.now() - startTime,
    };
  }
}

// ============================================
// 任务分类器类
// ============================================

export class TaskClassifier {
  private useAPI: boolean;
  private apiTimeout: number;
  private apiKey: string | undefined;
  
  constructor(options?: TaskClassifierOptions) {
    this.useAPI = options?.useAPI ?? false;
    this.apiTimeout = options?.apiTimeout ?? 2000;
    this.apiKey = process.env.DEEPBEEK_API_KEY;
  }
  
  /**
   * 单条文本分类
   */
  async classify(input: string): Promise<ClassificationResult> {
    const startTime = Date.now();
    
    if (this.useAPI && this.apiKey) {
      const apiResult = await classifyByAPI(input, this.apiKey, this.apiTimeout);
      if (apiResult.taskType !== 'unknown') {
        return apiResult;
      }
    }
    
    // 使用增强版规则模式
    const ruleResult = classifyByRulesEnhanced(input);
    return {
      ...ruleResult,
      method: 'rule',
      latencyMs: Date.now() - startTime,
    };
  }
  
  /**
   * 批量分类
   */
  async classifyBatch(inputs: string[]): Promise<ClassificationResult[]> {
    const promises = inputs.map(input => this.classify(input));
    return Promise.all(promises);
  }
  
  /**
   * 切换API模式
   */
  setUseAPI(useAPI: boolean): void {
    this.useAPI = useAPI;
  }
  
  /**
   * 设置API超时时间
   */
  setAPITimeout(timeout: number): void {
    this.apiTimeout = timeout;
  }
}

// 导出便捷函数
export async function classifyTask(input: string, useAPI = false): Promise<ClassificationResult> {
  const classifier = new TaskClassifier({ useAPI });
  return classifier.classify(input);
}

export async function classifyTaskBatch(inputs: string[], useAPI = false): Promise<ClassificationResult[]> {
  const classifier = new TaskClassifier({ useAPI });
  return classifier.classifyBatch(inputs);
}

// 导出增强版分类函数供测试使用
export { classifyByRulesEnhanced, calculateWeightedScore, detectCodeFeatures, detectMathFeatures, detectQAFeatures };
