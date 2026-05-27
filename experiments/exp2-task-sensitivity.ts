/**
 * 实验2：任务类型对压缩敏感性的差异
 * 假设：不同任务类型（数学推理/代码生成/对话QA）对KV压缩的敏感度不同
 */

import OpenAI from 'openai';

const apiKey = process.env.DEEPSEEK_API_KEY || 'sk-aec8f6c26a7048569e3819fdba235a08';
const client = new OpenAI({ apiKey, baseURL: 'https://api.deepseek.com' });

// 任务材料
const tasks = {
  math: [
    {
      context: `问题1：求解一元二次方程 x² - 5x + 6 = 0
解题步骤：
1. 识别系数：a=1, b=-5, c=6
2. 计算判别式：Δ = b² - 4ac = 25 - 24 = 1
3. 求根公式：x = (-b ± √Δ) / 2a
4. x₁ = (5 + 1) / 2 = 3
5. x₂ = (5 - 1) / 2 = 2

问题2：计算定积分 ∫₀² x² dx
解：利用幂函数积分公式 ∫ xⁿ dx = xⁿ⁺¹/(n+1) + C
∫₀² x² dx = [x³/3]₀² = 8/3 - 0 = 8/3`,
      question: '请解答：求方程 x² - 3x + 2 = 0 的根，并计算 ∫₁³ x² dx 的值。'
    },
    {
      context: `复利公式：A = P(1 + r/n)^(nt)
等差数列求和：S = n(a₁ + aₙ)/2
排列组合：C(n,k) = n! / (k!(n-k)!)
对数换底：logₐb = log b / log a
三角恒等式：sin²θ + cos²θ = 1
两点距离公式：d = √[(x₂-x₁)² + (y₂-y₁)²]`,
      question: '应用公式计算：如果本金P=10000，年利率r=5%，按季度复利计息，10年后的本息和是多少？（保留2位小数）'
    },
    {
      context: `向量点积：a·b = |a||b|cosθ = a₁b₁ + a₂b₂
向量叉积：|a×b| = |a||b|sinθ
矩阵乘法：C[i][j] = Σ A[i][k] * B[k][j]
行列式(2x2)：det = ad - bc
特征值方程：det(A - λI) = 0`,
      question: '已知向量a=(1,2), b=(3,4)，求它们的点积，并判断夹角是锐角、直角还是钝角。'
    }
  ],
  code: [
    {
      context: `Python排序API文档摘要：
- sorted(iterable, *, key=None, reverse=False) -> list
  返回按升序排列的新列表
- list.sort(*, key=None, reverse=False)  
  就地排序，不返回新列表
- heapq.heapify(x) 将列表转换为堆
- heapq.heappush(heap, item) 推入堆
- heapq.heappop(heap) 弹出堆最小值

参数说明：
- key: 排序依据的函数，如 key=lambda x: x['age']
- reverse: True时降序，False时升序`,
      question: '写一个Python函数，接收包含"name"和"score"的学生字典列表，按score降序排列，相同score按name升序排列。'
    },
    {
      context: `JavaScript数组方法速查：
- Array.map(fn) -> 返回转换后的新数组
- Array.filter(fn) -> 返回满足条件的新数组  
- Array.reduce(fn, init) -> 返回聚合结果
- Array.forEach(fn) -> 遍历，无返回值
- Array.find(fn) -> 返回第一个满足条件的元素
- Array.some(fn) -> 返回布尔值
- Array.every(fn) -> 检查是否全部满足`,
      question: '用map/filter/reduce实现：给定数字数组[1,2,3,4,5,6,7,8,9,10]，返回所有偶数的平方和。'
    },
    {
      context: `Git常用命令参考：
- git init 初始化仓库
- git add <file> 暂存文件
- git commit -m "msg" 提交
- git status 查看状态
- git log --oneline 查看简短日志
- git branch <name> 创建分支
- git checkout <branch> 切换分支
- git merge <branch> 合并分支
- git rebase <base> 变基操作`,
      question: '场景：你在feature分支开发，main分支有新提交。请写出将main分支合并到feature分支的完整步骤。'
    }
  ],
  qa: [
    {
      context: `对话历史：
用户：我想要一个手机推荐
助手：根据您的需求（拍照、游戏、续航），我推荐以下几款...（列出详细对比）

用户：拍照最重要，其他可以妥协
助手：明白，侧重拍照的话，建议考虑...（给出具体建议）

用户：预算5000以内
助手：好的，这个预算内拍照最好的选择是...`,
      question: '请根据对话历史，总结用户最终的手机需求，并给出推荐理由。'
    },
    {
      context: `系统提示：你是一个专业的旅行规划助手。
用户：五一假期想去云南玩，3-4天
助手：云南推荐行程：
- 丽江古城+玉龙雪山（2天）
- 大理洱海+古城（1-2天）
- 泸沽湖（可选，需额外1天）

用户：主要想放松，不赶景点
助手：建议精简行程，丽江+大理即可...（详细建议）

用户：适合情侣吗？
助手：非常适合！推荐...（浪漫目的地）`,
      question: '作为旅行规划助手，根据以上对话，给出一个适合这对情侣的4天云南行程规划，包含每天的具体安排。'
    },
    {
      context: `技术问题上下文：
问题描述：用户反馈API接口响应慢
排查过程：
- 检查了服务器CPU/内存，正常
- 查看数据库查询日志，发现有几个查询超过5秒
- 网络延迟测试，本地到服务器50ms，正常范围
- 检查到某表缺少索引，该表数据量500万+

初步结论：缺少索引导致全表扫描
建议方案：添加复合索引...`,
      question: '作为技术支持工程师，根据以上排查记录，写一份问题分析报告，包含原因分析和后续优化建议。'
    }
  ]
};

// 压缩函数
function compressMiddle(text: string): string {
  const lines = text.split('\n');
  const mid = Math.floor(lines.length / 2);
  return lines.slice(0, mid).join('\n') + '\n[...中间内容已压缩...]\n' + lines.slice(-1).join('\n');
}

function compressHead(text: string): string {
  const lines = text.split('\n');
  const keepLines = Math.max(1, Math.floor(lines.length * 0.3));
  return `[上文已压缩]\n` + lines.slice(-keepLines).join('\n');
}

// 评分函数
function scoreResponse(response: string, taskType: string): number {
  let base = 5;
  
  if (taskType === 'math') {
    if (response.match(/\d+(\.\d+)?/g)) base += 2;
    if (response.includes('解') || response.includes('答案')) base += 1;
    if (response.length > 100) base += 1;
  } else if (taskType === 'code') {
    if (response.includes('def ') || response.includes('function')) base += 2;
    if (response.includes('return')) base += 1;
    if (response.includes('for') || response.includes('while')) base += 1;
  } else if (taskType === 'qa') {
    if (response.includes('建议') || response.includes('推荐')) base += 1;
    if (response.includes('根据') || response.includes('基于')) base += 1;
    if (response.length > 80) base += 1;
  }
  
  return Math.min(10, base);
}

async function runTask(
  task: any,
  taskType: string,
  compression: string
): Promise<any> {
  const context = compression === 'middle' ? compressMiddle(task.context) :
                  compression === 'head' ? compressHead(task.context) :
                  task.context;
  
  const prompt = `【上下文】\n${context}\n\n【问题】\n${task.question}`;
  
  const startTime = Date.now();
  
  try {
    const response = await client.chat.completions.create({
      model: 'deepseek-chat',
      messages: [{ role: 'user', content: prompt }],
      max_tokens: 500,
      temperature: 0.7
    });
    
    const ttft = Date.now() - startTime;
    const content = response.choices[0]?.message?.content || '';
    const score = scoreResponse(content, taskType);
    
    return { taskType, compression, score, ttft, content: content.slice(0, 80) };
  } catch (error: any) {
    console.error(`Error (${taskType}/${compression}):`, error.message);
    return { taskType, compression, score: 0, ttft: 0, error: error.message };
  }
}

function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

async function main() {
  console.log('========================================');
  console.log('实验2：任务类型对压缩敏感性的差异');
  console.log('========================================\n');
  
  const results: any[] = [];
  const taskTypes = ['math', 'code', 'qa'];
  const compressions = ['none', 'middle', 'head'];
  
  for (const taskType of taskTypes) {
    console.log(`\n=== 任务类型: ${taskType.toUpperCase()} ===`);
    
    for (const task of tasks[taskType as keyof typeof tasks]) {
      for (const compression of compressions) {
        console.log(`  运行 ${compression} 压缩...`);
        const result = await runTask(task, taskType, compression);
        results.push(result);
        
        if (!result.error) {
          console.log(`    质量=${result.score}, TTFT=${result.ttft}ms`);
        }
        
        await sleep(1500);
      }
    }
  }
  
  // 汇总
  console.log('\n\n========================================');
  console.log('实验结果汇总');
  console.log('========================================');
  
  console.log('\n| 任务类型 | 压缩方式 | 平均质量 | 平均TTFT(ms) |');
  console.log('|----------|----------|----------|--------------|');
  
  const summary: any = {};
  
  for (const taskType of taskTypes) {
    summary[taskType] = {};
    
    for (const compression of compressions) {
      const filtered = results.filter(r => 
        r.taskType === taskType && r.compression === compression && !r.error
      );
      
      if (filtered.length > 0) {
        const avgScore = filtered.reduce((s, r) => s + r.score, 0) / filtered.length;
        const avgTtft = filtered.reduce((s, r) => s + r.ttft, 0) / filtered.length;
        
        summary[taskType][compression] = { avgScore, avgTtft };
        console.log(`| ${taskType} | ${compression} | ${avgScore.toFixed(2)} | ${avgTtft.toFixed(0)} |`);
      }
    }
  }
  
  // 敏感度分析
  console.log('\n\n压缩敏感度分析:');
  for (const taskType of taskTypes) {
    const baseline = summary[taskType]?.none?.avgScore || 0;
    if (baseline > 0) {
      const middleLoss = ((baseline - (summary[taskType]?.middle?.avgScore || 0)) / baseline * 100).toFixed(1);
      const headLoss = ((baseline - (summary[taskType]?.head?.avgScore || 0)) / baseline * 100).toFixed(1);
      console.log(`  ${taskType}: 中间压缩损失=${middleLoss}%, 头部压缩损失=${headLoss}%`);
    }
  }
  
  return results;
}

main().then(() => console.log('\n实验2完成！')).catch(console.error);
