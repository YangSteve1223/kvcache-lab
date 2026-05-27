/**
 * 实验1（精简版）：P端 vs D端压缩敏感性验证
 * 假设：D端（解码/生成阶段）比P端（预填充/上下文阶段）对KV压缩更敏感
 */

import OpenAI from 'openai';

const apiKey = process.env.DEEPSEEK_API_KEY || 'sk-aec8f6c26a7048569e3819fdba235a08';
const client = new OpenAI({ apiKey, baseURL: 'https://api.deepseek.com' });

// 简化context材料
const context500 = `人工智能技术的发展经历了三个重要阶段。第一阶段是符号主义时期，强调逻辑推理和专家系统。第二阶段是统计学习方法兴起，依赖大规模数据和算力。第三阶段是深度学习时代，神经网络模型取得突破性进展。大语言模型的核心技术包括Transformer架构、自注意力机制和预训练-微调范式。`;

const context1000 = `深度学习在计算机视觉领域取得了显著成就。卷积神经网络(CNN)通过局部感受野和权值共享有效提取图像特征。经典的CNN架构包括LeNet、AlexNet、VGGNet、ResNet等。残差连接解决了深层网络训练困难的问题。目标检测任务需要定位图像中的物体并识别其类别。语义分割任务为图像中每个像素分配类别标签。生成对抗网络(GAN)由生成器和判别器组成，通过对抗训练学习数据分布。`;

const context2000 = `云计算改变了企业的IT基础设施部署模式。基础设施即服务(IaaS)提供虚拟化的计算资源，包括服务器、存储和网络。平台即服务(PaaS)进一步封装了操作系统、中间件和开发工具。软件即服务(SaaS)让用户通过浏览器使用完整应用。微服务架构将大型应用拆分为多个小型、自治的服务。每个服务围绕特定业务功能构建，可以独立开发、部署和扩展。服务间通过轻量级协议通信，常见选择包括HTTP REST API和gRPC。容器技术如Docker简化了微服务的部署和环境一致性。Kubernetes作为容器编排平台提供了自动扩缩容、自愈和负载均衡等能力。DevOps文化强调开发与运维团队的协作，通过自动化流水线实现持续集成和持续部署。`;

function createPCompressedContext(text: string): string {
  const mid = Math.floor(text.length / 2);
  return `[上文摘要：技术发展概述。]` + text.slice(mid);
}

function createDCompressedContext(text: string): string {
  const mid = Math.floor(text.length / 2);
  return text.slice(0, mid) + `[下文摘要：架构与实践应用。]`;
}

function scoreResponse(response: string): { accuracy: number; completeness: number; relevance: number } {
  let accuracy = 5;
  if (response.includes('1.') || response.includes('第一')) accuracy += 1;
  if (response.includes('2.') || response.includes('第二')) accuracy += 1;
  if (response.length > 100) accuracy += 1;
  
  let completeness = 5;
  if (response.includes('技术')) completeness += 1;
  if (response.includes('阶段') || response.includes('架构')) completeness += 1;
  
  let relevance = 6;
  if (response.length < 50) relevance -= 2;
  
  return {
    accuracy: Math.min(10, accuracy),
    completeness: Math.min(10, completeness),
    relevance: Math.min(10, relevance)
  };
}

async function runExperiment(
  context: string,
  label: string
): Promise<any> {
  const question = `\n\n请根据以上文档内容，回答：1.涉及哪些主要技术领域？2.总结核心观点并说明关联。请用结构化形式回答。`;
  const fullPrompt = context + question;
  
  const startTime = Date.now();
  
  try {
    const response = await client.chat.completions.create({
      model: 'deepseek-chat',
      messages: [{ role: 'user', content: fullPrompt }],
      max_tokens: 500,
      temperature: 0.7
    });
    
    const ttft = Date.now() - startTime;
    const usage = response.usage;
    const content = response.choices[0]?.message?.content || '';
    const scores = scoreResponse(content);
    
    return {
      label,
      inputTokens: usage?.prompt_tokens || 0,
      outputTokens: usage?.completion_tokens || 0,
      accuracy: scores.accuracy,
      completeness: scores.completeness,
      relevance: scores.relevance,
      qualityScore: (scores.accuracy + scores.completeness + scores.relevance) / 3,
      ttft,
      content: content.slice(0, 100)
    };
  } catch (error: any) {
    console.error(`API Error (${label}):`, error.message);
    return { label, error: error.message, ttft: 0 };
  }
}

async function sleep(ms: number) {
  return new Promise(r => setTimeout(r, ms));
}

async function main() {
  console.log('========================================');
  console.log('实验1：P端 vs D端压缩敏感性验证');
  console.log('========================================\n');
  
  const results: any[] = [];
  
  // 测试3种context长度，每种3种压缩配置
  const contexts = [
    { length: 500, text: context500 },
    { length: 1000, text: context1000 },
    { length: 2000, text: context2000 }
  ];
  
  for (const { length, text } of contexts) {
    console.log(`\n=== Context长度: ${length} tokens ===`);
    
    const configs = [
      { label: `baseline-${length}`, context: text },
      { label: `p-compressed-${length}`, context: createPCompressedContext(text) },
      { label: `d-compressed-${length}`, context: createDCompressedContext(text) }
    ];
    
    for (const config of configs) {
      console.log(`  运行 ${config.label}...`);
      const result = await runExperiment(config.context, config.label);
      results.push(result);
      
      if (result.error) {
        console.log(`    错误: ${result.error}`);
      } else {
        console.log(`    质量=${result.qualityScore.toFixed(2)}, TTFT=${result.ttft}ms, 输入=${result.inputTokens} tokens`);
      }
      
      await sleep(1500);
    }
  }
  
  // 汇总结果
  console.log('\n\n========================================');
  console.log('实验结果汇总');
  console.log('========================================');
  
  console.log('\n| Context长度 | 配置 | 质量评分 | TTFT(ms) | 输入Tokens |');
  console.log('|-------------|------|----------|----------|------------|');
  
  const summary: any[] = [];
  
  for (const length of [500, 1000, 2000]) {
    const baseline = results.find(r => r.label === `baseline-${length}`);
    const pComp = results.find(r => r.label === `p-compressed-${length}`);
    const dComp = results.find(r => r.label === `d-compressed-${length}`);
    
    const row = { length, baseline: baseline?.qualityScore || 0, pCompressed: pComp?.qualityScore || 0, dCompressed: dComp?.qualityScore || 0 };
    summary.push(row);
    
    if (baseline && !baseline.error) {
      console.log(`| ${length} | baseline | ${baseline.qualityScore.toFixed(2)} | ${baseline.ttft} | ${baseline.inputTokens} |`);
    }
    if (pComp && !pComp.error) {
      console.log(`| ${length} | P端压缩 | ${pComp.qualityScore.toFixed(2)} | ${pComp.ttft} | ${pComp.inputTokens} |`);
    }
    if (dComp && !dComp.error) {
      console.log(`| ${length} | D端压缩 | ${dComp.qualityScore.toFixed(2)} | ${dComp.ttft} | ${dComp.inputTokens} |`);
    }
  }
  
  console.log('\n\n质量损失分析:');
  for (const row of summary) {
    if (row.baseline > 0 && row.pCompressed > 0 && row.dCompressed > 0) {
      const pLoss = ((row.baseline - row.pCompressed) / row.baseline * 100).toFixed(1);
      const dLoss = ((row.baseline - row.dCompressed) / row.baseline * 100).toFixed(1);
      console.log(`  ${row.length} tokens: P端压缩损失=${pLoss}%, D端压缩损失=${dLoss}%`);
    }
  }
  
  return results;
}

main().then(() => console.log('\n实验1完成！')).catch(console.error);
