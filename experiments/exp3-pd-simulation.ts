/**
 * 实验3：PD分离+压缩联合仿真
 * 基于LLM-Serving-Lab的PD模拟器逻辑，实现简化版仿真
 */

interface Request {
  id: number;
  inputTokens: number;
  outputTokens: number;
  type: 'short' | 'medium' | 'long'; // 影响batch优先级
}

interface SimConfig {
  name: string;
  prefillCompression: number; // 0-1, 1表示无压缩
  decodeCompression: number;
}

interface SimResult {
  config: string;
  avgTTFT: number;
  avgE2ELatency: number;
  throughput: number; // tokens/sec
}

// 仿真参数
const BASE_MS = 10; // 基础延迟 ms
const MS_PER_INPUT_TOKEN = 0.1; // prefill每token耗时
const MS_PER_OUTPUT_TOKEN = 0.5; // decode每token耗时
const KV_SIZE_KB = 64; // 单token KV大小(KB)
const BANDWIDTH_GBPS = 100; // KV传输带宽

// 仿真函数
function simulatePD(
  requests: Request[],
  config: SimConfig
): SimResult {
  let totalTTFT = 0;
  let totalE2E = 0;
  let totalOutputTokens = 0;
  
  // 简化的串行处理模拟
  let currentTime = 0;
  
  for (const req of requests) {
    // Prefill阶段
    // 压缩后的有效输入tokens
    const effectiveInputTokens = req.inputTokens * config.prefillCompression;
    const prefillTime = BASE_MS + effectiveInputTokens * MS_PER_INPUT_TOKEN;
    
    // KV传输时间（decode阶段需要传输KV cache）
    const kvTransferTime = (KV_SIZE_KB * req.inputTokens) / (BANDWIDTH_GBPS * 1000 * 1000) * 1000 * (1 - config.decodeCompression);
    
    // TTFT = prefill时间 + KV传输时间
    const ttft = prefillTime + kvTransferTime;
    
    // Decode阶段
    const decodeTime = BASE_MS + req.outputTokens * MS_PER_OUTPUT_TOKEN;
    
    // E2E延迟 = TTFT + decode时间
    const e2e = ttft + decodeTime;
    
    totalTTFT += ttft;
    totalE2E += e2e;
    totalOutputTokens += req.outputTokens;
    
    currentTime += prefillTime + decodeTime;
  }
  
  const n = requests.length;
  const avgTTFT = totalTTFT / n;
  const avgE2ELatency = totalE2E / n;
  const throughput = (totalOutputTokens / currentTime) * 1000; // tokens/sec
  
  return {
    config: config.name,
    avgTTFT: Math.round(avgTTFT * 100) / 100,
    avgE2ELatency: Math.round(avgE2ELatency * 100) / 100,
    throughput: Math.round(throughput * 100) / 100
  };
}

// 生成模拟请求序列
function generateRequests(count: number): Request[] {
  const requests: Request[] = [];
  const types: Array<'short' | 'medium' | 'long'> = ['short', 'medium', 'long'];
  
  for (let i = 0; i < count; i++) {
    const type = types[i % 3];
    let inputTokens: number, outputTokens: number;
    
    switch (type) {
      case 'short':
        inputTokens = 100 + Math.floor(Math.random() * 200);
        outputTokens = 50 + Math.floor(Math.random() * 100);
        break;
      case 'medium':
        inputTokens = 500 + Math.floor(Math.random() * 500);
        outputTokens = 200 + Math.floor(Math.random() * 300);
        break;
      case 'long':
        inputTokens = 2000 + Math.floor(Math.random() * 1000);
        outputTokens = 500 + Math.floor(Math.random() * 500);
        break;
    }
    
    requests.push({ id: i, inputTokens, outputTokens, type });
  }
  
  return requests;
}

async function main() {
  console.log('========================================');
  console.log('实验3：PD分离+压缩联合仿真');
  console.log('========================================\n');
  
  console.log('仿真参数:');
  console.log(`  基础延迟: ${BASE_MS}ms`);
  console.log(`  Prefill每token: ${MS_PER_INPUT_TOKEN}ms`);
  console.log(`  Decode每token: ${MS_PER_OUTPUT_TOKEN}ms`);
  console.log(`  KV大小/token: ${KV_SIZE_KB}KB`);
  console.log(`  传输带宽: ${BANDWIDTH_GBPS}GB/s`);
  console.log('');
  
  // 生成请求序列
  const requests = generateRequests(30);
  
  console.log('请求序列统计:');
  const shortReqs = requests.filter(r => r.type === 'short');
  const mediumReqs = requests.filter(r => r.type === 'medium');
  const longReqs = requests.filter(r => r.type === 'long');
  
  console.log(`  短请求: ${shortReqs.length}个, 平均输入=${(shortReqs.reduce((s,r)=>s+r.inputTokens,0)/shortReqs.length).toFixed(0)} tokens`);
  console.log(`  中请求: ${mediumReqs.length}个, 平均输入=${(mediumReqs.reduce((s,r)=>s+r.inputTokens,0)/mediumReqs.length).toFixed(0)} tokens`);
  console.log(`  长请求: ${longReqs.length}个, 平均输入=${(longReqs.reduce((s,r)=>s+r.inputTokens,0)/longReqs.length).toFixed(0)} tokens`);
  console.log('');
  
  // 3种配置
  const configs: SimConfig[] = [
    { name: 'No compression', prefillCompression: 1.0, decodeCompression: 1.0 },
    { name: 'Uniform compression (0.5)', prefillCompression: 0.5, decodeCompression: 0.5 },
    { name: 'PD-Aware (P:0.3, D:0.7)', prefillCompression: 0.3, decodeCompression: 0.7 }
  ];
  
  console.log('========================================');
  console.log('仿真结果');
  console.log('========================================\n');
  
  const results: SimResult[] = [];
  
  for (const config of configs) {
    console.log(`配置: ${config.name}`);
    console.log(`  Prefill压缩比: ${config.prefillCompression}, Decode压缩比: ${config.decodeCompression}`);
    
    const result = simulatePD(requests, config);
    results.push(result);
    
    console.log(`  平均TTFT: ${result.avgTTFT}ms`);
    console.log(`  平均E2E延迟: ${result.avgE2ELatency}ms`);
    console.log(`  吞吐量: ${result.throughput} tokens/sec`);
    console.log('');
  }
  
  // 结果对比表格
  console.log('========================================');
  console.log('配置对比汇总');
  console.log('========================================');
  
  console.log('\n| 配置 | 平均TTFT(ms) | 平均E2E(ms) | 吞吐量(tokens/s) |');
  console.log('|------|-------------|-------------|-----------------|');
  
  for (const r of results) {
    console.log(`| ${r.config} | ${r.avgTTFT} | ${r.avgE2ELatency} | ${r.throughput} |`);
  }
  
  // 分析
  console.log('\n\n性能提升分析:');
  
  const baseline = results[0];
  const uniform = results[1];
  const pdAware = results[2];
  
  const ttftImprovementUniform = ((baseline.avgTTFT - uniform.avgTTFT) / baseline.avgTTFT * 100).toFixed(1);
  const ttftImprovementPDAware = ((baseline.avgTTFT - pdAware.avgTTFT) / baseline.avgTTFT * 100).toFixed(1);
  
  const e2eImprovementUniform = ((baseline.avgE2ELatency - uniform.avgE2ELatency) / baseline.avgE2ELatency * 100).toFixed(1);
  const e2eImprovementPDAware = ((baseline.avgE2ELatency - pdAware.avgE2ELatency) / baseline.avgE2ELatency * 100).toFixed(1);
  
  const tpImprovementUniform = ((uniform.throughput - baseline.throughput) / baseline.throughput * 100).toFixed(1);
  const tpImprovementPDAware = ((pdAware.throughput - baseline.throughput) / baseline.throughput * 100).toFixed(1);
  
  console.log(`\nvs 无压缩 baseline:`);
  console.log(`  Uniform压缩: TTFT改善${ttftImprovementUniform}%, E2E改善${e2eImprovementUniform}%, 吞吐提升${tpImprovementUniform}%`);
  console.log(`  PD-Aware:   TTFT改善${ttftImprovementPDAware}%, E2E改善${e2eImprovementPDAware}%, 吞吐提升${tpImprovementPDAware}%`);
  
  console.log('\n\n关键发现:');
  if (parseFloat(ttftImprovementPDAware) > parseFloat(ttftImprovementUniform)) {
    console.log('  ✓ PD-Aware压缩策略在TTFT上优于Uniform压缩');
  } else {
    console.log('  ✗ Uniform压缩在TTFT上优于PD-Aware压缩');
  }
  
  if (parseFloat(e2eImprovementPDAware) > parseFloat(e2eImprovementUniform)) {
    console.log('  ✓ PD-Aware压缩策略在E2E延迟上优于Uniform压缩');
  } else {
    console.log('  ✗ Uniform压缩在E2E延迟上优于PD-Aware压缩');
  }
  
  return results;
}

main().then(() => console.log('\n实验3完成！')).catch(console.error);
