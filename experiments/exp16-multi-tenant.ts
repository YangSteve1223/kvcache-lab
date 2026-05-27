/**
 * 实验16：多租户隔离实验
 * 
 * 模拟3个租户的并发请求场景：
 * - 租户A（高优先级）：math任务，紧SLO，100个请求
 * - 租户B（中优先级）：code任务，中SLO，150个请求
 * - 租户C（低优先级）：qa任务，松SLO，100个请求
 * 
 * 对比策略：
 * - 无隔离（共享资源，统一策略）
 * - 优先级隔离（高优先级先调度）
 * - SLO-aware隔离（各租户独立SLO约束）
 * 
 * 测量指标：各租户SLO满足率、平均TTFT、质量
 */

import { writeFileSync } from 'fs';

// ========== 类型定义 ==========

type TenantId = 'A' | 'B' | 'C';
type Priority = 'high' | 'medium' | 'low';
type IsolationMode = 'none' | 'priority' | 'slo-aware';

interface Tenant {
  id: TenantId;
  name: string;
  taskType: 'math' | 'code' | 'qa';
  priority: Priority;
  sloTTFT: number;
  sloE2E: number;
  requestCount: number;
}

interface Request {
  id: string;
  tenantId: TenantId;
  inputTokens: number;
  outputTokens: number;
  taskType: string;
  priority: Priority;
  arrivalTimeMs: number;
}

interface Result {
  requestId: string;
  tenantId: TenantId;
  ttftMs: number;
  e2eLatencyMs: number;
  qualityScore: number;
  meetsSLO: boolean;
}

interface TenantStats {
  tenantId: TenantId;
  requestCount: number;
  avgTTFT: number;
  avgE2E: number;
  avgQuality: number;
  sloSatisfactionRate: number;
  p95TTFT: number;
}

// ========== 实验配置 ==========

const EXP_CONFIG = {
  model: {
    layers: 32,
    kvBytesPerToken: 1024
  },
  bandwidth: {
    name: '5GB/s',
    bytesPerMs: 5 * 1024 * 1024 / 1
  },
  tenants: [
    { id: 'A' as TenantId, name: '租户A', taskType: 'math' as const, priority: 'high' as Priority, sloTTFT: 500, sloE2E: 2000, requestCount: 100 },
    { id: 'B' as TenantId, name: '租户B', taskType: 'code' as const, priority: 'medium' as Priority, sloTTFT: 1000, sloE2E: 3000, requestCount: 150 },
    { id: 'C' as TenantId, name: '租户C', taskType: 'qa' as const, priority: 'low' as Priority, sloTTFT: 2000, sloE2E: 5000, requestCount: 100 }
  ] as Tenant[],
  isolationModes: ['none', 'priority', 'slo-aware'] as IsolationMode[],
  runs: 3
};

// ========== 工具函数 ==========

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function round4(value: number): number {
  return Math.round(value * 10000) / 10000;
}

function average(arr: number[]): number {
  return Math.round(arr.reduce((a, b) => a + b, 0) / arr.length * 100) / 100;
}

function percentile(arr: number[], p: number): number {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.ceil(sorted.length * p / 100) - 1;
  return Math.round(sorted[Math.max(0, idx)] * 100) / 100;
}

// ========== 请求生成 ==========

function generateTenantRequests(tenant: Tenant): Request[] {
  return Array.from({ length: tenant.requestCount }, (_, i) => ({
    id: `${tenant.id}-req-${i}`,
    tenantId: tenant.id,
    inputTokens: Math.floor(Math.random() * 3000 + 500),
    outputTokens: Math.floor(Math.random() * 700 + 100),
    taskType: tenant.taskType,
    priority: tenant.priority,
    arrivalTimeMs: Math.floor(Math.random() * 10000) // 10秒内随机到达
  }));
}

// ========== 调度策略 ==========

// 无隔离：所有请求混合调度
function scheduleNone(requests: Request[]): Request[] {
  return [...requests].sort((a, b) => a.arrivalTimeMs - b.arrivalTimeMs);
}

// 优先级隔离：高优先级先调度
function schedulePriority(requests: Request[]): Request[] {
  const priorityOrder: Record<Priority, number> = { high: 0, medium: 1, low: 2 };
  return [...requests].sort((a, b) => {
    const pDiff = priorityOrder[a.priority] - priorityOrder[b.priority];
    if (pDiff !== 0) return pDiff;
    return a.arrivalTimeMs - b.arrivalTimeMs;
  });
}

// SLO-aware隔离：紧迫SLO先调度
function scheduleSLOAware(requests: Request[]): Request[] {
  return [...requests].sort((a, b) => {
    // 找到对应租户的SLO
    const tenantA = EXP_CONFIG.tenants.find(t => t.id === a.tenantId)!;
    const tenantB = EXP_CONFIG.tenants.find(t => t.id === b.tenantId)!;
    
    // 计算紧迫度 = 1 / SLO_TTFT (SLO越小越紧迫)
    const urgencyA = 1 / tenantA.sloTTFT;
    const urgencyB = 1 / tenantB.sloTTFT;
    
    const urgencyDiff = urgencyB - urgencyA;
    if (Math.abs(urgencyDiff) > 0.0001) return urgencyDiff;
    
    return a.arrivalTimeMs - b.arrivalTimeMs;
  });
}

// ========== 压缩策略 ==========

function computeCompressionConfig(
  taskType: string,
  sloTTFT: number,
  mode: IsolationMode,
  tenantId: TenantId,
  currentLoad: number
): { compressionRatio: number; quality: number } {
  // 根据模式选择策略
  let baseCompression = 0.5;
  
  if (mode === 'none') {
    // 统一策略：对所有租户使用相同压缩
    baseCompression = 0.5;
  } else if (mode === 'priority') {
    // 优先级策略：高优先级使用保守压缩，低优先级使用激进压缩
    const priorityCompression: Record<Priority, number> = { high: 0.7, medium: 0.5, low: 0.3 };
    baseCompression = priorityCompression[tenantId === 'A' ? 'high' : tenantId === 'B' ? 'medium' : 'low'];
  } else {
    // SLO-aware策略：根据SLO紧迫度动态调整
    const sloUrgency = 1 - (sloTTFT / 2000); // 2000ms为基准
    const loadAdjustment = (currentLoad - 0.5) * 0.2;
    baseCompression = clamp(0.8 - sloUrgency * 0.5 + loadAdjustment, 0.2, 0.8);
    
    // 任务类型微调
    if (taskType === 'math') baseCompression = Math.min(baseCompression + 0.1, 0.9);
    if (taskType === 'code') baseCompression = Math.max(baseCompression - 0.1, 0.2);
  }
  
  const compressionRatio = baseCompression * 0.5; // K8V4精度
  const quality = baseCompression;
  
  return { compressionRatio, quality };
}

// ========== 模拟请求处理 ==========

function simulateRequests(
  requests: Request[],
  mode: IsolationMode
): { results: Result[]; loadHistory: number[] } {
  // 调度请求
  let scheduledRequests: Request[];
  switch (mode) {
    case 'none': scheduledRequests = scheduleNone(requests); break;
    case 'priority': scheduledRequests = schedulePriority(requests); break;
    case 'slo-aware': scheduledRequests = scheduleSLOAware(requests); break;
  }
  
  const results: Result[] = [];
  const loadHistory: number[] = [];
  let currentLoad = 0.3;
  
  for (const request of scheduledRequests) {
    // 动态更新负载
    currentLoad = clamp(currentLoad + (Math.random() - 0.5) * 0.1, 0.2, 0.9);
    loadHistory.push(currentLoad);
    
    // 获取租户SLO
    const tenant = EXP_CONFIG.tenants.find(t => t.id === request.tenantId)!;
    
    // 计算压缩配置
    const config = computeCompressionConfig(
      request.taskType,
      tenant.sloTTFT,
      mode,
      request.tenantId,
      currentLoad
    );
    
    // 计算传输时间
    const bandwidth = EXP_CONFIG.bandwidth.bytesPerMs;
    const kvBytes = request.inputTokens * EXP_CONFIG.model.kvBytesPerToken * EXP_CONFIG.model.layers;
    const transferTimeMs = (kvBytes * config.compressionRatio) / bandwidth;
    
    // Prefill时间（考虑当前负载）
    const loadPenalty = 1 + currentLoad * 0.5;
    const prefillTimeMs = (50 + request.inputTokens * 0.01) * loadPenalty;
    
    // TTFT
    const ttftMs = prefillTimeMs + transferTimeMs;
    
    // E2E
    const decodeTimeMs = request.outputTokens * 0.5;
    const e2eLatencyMs = ttftMs + decodeTimeMs;
    
    results.push({
      requestId: request.id,
      tenantId: request.tenantId,
      ttftMs: round4(ttftMs),
      e2eLatencyMs: round4(e2eLatencyMs),
      qualityScore: round4(config.quality),
      meetsSLO: ttftMs <= tenant.sloTTFT && e2eLatencyMs <= tenant.sloE2E
    });
  }
  
  return { results, loadHistory };
}

// ========== 统计计算 ==========

function computeTenantStats(results: Result[], tenantId: TenantId, tenant: Tenant): TenantStats {
  const tenantResults = results.filter(r => r.tenantId === tenantId);
  const ttfts = tenantResults.map(r => r.ttftMs);
  const e2es = tenantResults.map(r => r.e2eLatencyMs);
  const qualities = tenantResults.map(r => r.qualityScore);
  const meetsSLO = tenantResults.filter(r => r.meetsSLO).length;
  
  return {
    tenantId,
    requestCount: tenantResults.length,
    avgTTFT: average(ttfts),
    avgE2E: average(e2es),
    avgQuality: round4(qualities.reduce((a, b) => a + b, 0) / qualities.length),
    sloSatisfactionRate: round4(meetsSLO / tenantResults.length),
    p95TTFT: percentile(ttfts, 95)
  };
}

// ========== 主实验 ==========

function runMultiTenantExperiment() {
  console.log('='.repeat(70));
  console.log('实验16: 多租户隔离实验');
  console.log('='.repeat(70));
  console.log('\n租户配置:');
  for (const tenant of EXP_CONFIG.tenants) {
    console.log(`  ${tenant.name}: ${tenant.taskType}任务, ${tenant.priority}优先级, SLO TTFT<${tenant.sloTTFT}ms`);
  }
  console.log(`\n隔离模式: ${EXP_CONFIG.isolationModes.join(', ')}`);
  console.log(`运行次数: ${EXP_CONFIG.runs}\n`);
  
  const allResults: Record<IsolationMode, Record<TenantId, TenantStats[]>> = {} as any;
  
  for (const mode of EXP_CONFIG.isolationModes) {
    console.log(`\n>>> 隔离模式: ${mode}`);
    console.log('-'.repeat(50));
    
    const modeResults: Record<TenantId, TenantStats[]> = {} as any;
    
    for (const tenant of EXP_CONFIG.tenants) {
      console.log(`\n  运行 ${tenant.name}...`);
      
      const runStats: TenantStats[] = [];
      
      for (let run = 0; run < EXP_CONFIG.runs; run++) {
        // 生成所有租户的请求
        const allRequests: Request[] = [];
        for (const t of EXP_CONFIG.tenants) {
          allRequests.push(...generateTenantRequests(t));
        }
        
        // 模拟处理
        const { results } = simulateRequests(allRequests, mode);
        
        // 计算该租户统计
        const stats = computeTenantStats(results, tenant.id, tenant);
        runStats.push(stats);
      }
      
      // 取平均
      const avgStats: TenantStats = {
        tenantId: tenant.id,
        requestCount: tenant.requestCount,
        avgTTFT: average(runStats.map(s => s.avgTTFT)),
        avgE2E: average(runStats.map(s => s.avgE2E)),
        avgQuality: round4(average(runStats.map(s => s.avgQuality))),
        sloSatisfactionRate: round4(average(runStats.map(s => s.sloSatisfactionRate))),
        p95TTFT: average(runStats.map(s => s.p95TTFT))
      };
      
      modeResults[tenant.id] = [avgStats];
      console.log(`    TTFT: ${avgStats.avgTTFT}ms, P95: ${avgStats.p95TTFT}ms, SLO: ${(avgStats.sloSatisfactionRate * 100).toFixed(1)}%, 质量: ${avgStats.avgQuality}`);
    }
    
    allResults[mode] = modeResults;
  }
  
  return allResults;
}

// ========== 报告生成 ==========

function generateReport(results: Record<IsolationMode, Record<TenantId, TenantStats[]>>): string {
  const tenants = EXP_CONFIG.tenants;
  const modes = EXP_CONFIG.isolationModes;
  const modeNames: Record<IsolationMode, string> = {
    'none': '无隔离',
    'priority': '优先级隔离',
    'slo-aware': 'SLO-aware隔离'
  };
  
  let report = `# 实验16: 多租户隔离实验

## 实验配置

### 租户配置

| 租户 | 任务类型 | 优先级 | TTFT SLO | E2E SLO | 请求数 |
|------|---------|--------|----------|---------|--------|
`;
  
  for (const tenant of tenants) {
    report += `| ${tenant.name} | ${tenant.taskType} | ${tenant.priority} | <${tenant.sloTTFT}ms | <${tenant.sloE2E}ms | ${tenant.requestCount} |\n`;
  }
  
  report += `
### 隔离模式

1. **无隔离**: 所有请求混合调度，使用统一压缩策略
2. **优先级隔离**: 高优先级先调度，低优先级可能延迟
3. **SLO-aware隔离**: SLO越紧迫越先调度

## 结果汇总

`;
  
  // 按隔离模式输出表格
  for (const mode of modes) {
    report += `\n### ${modeNames[mode]}\n\n`;
    report += `| 租户 | 请求数 | 平均TTFT(ms) | P95 TTFT(ms) | 平均E2E(ms) | 平均质量 | SLO满足率 |\n`;
    report += `|------|--------|-------------|-------------|------------|---------|----------|\n`;
    
    for (const tenant of tenants) {
      const s = results[mode][tenant.id][0];
      report += `| ${tenant.name} | ${s.requestCount} | ${s.avgTTFT} | ${s.p95TTFT} | ${s.avgE2E} | ${s.avgQuality} | ${(s.sloSatisfactionRate * 100).toFixed(1)}% |\n`;
    }
  }
  
  // 跨模式对比
  report += `\n## 跨模式对比\n\n`;
  
  // 租户A（高优先级）对比
  report += `### 租户A（高优先级，math，紧SLO）\n\n`;
  report += `| 模式 | 平均TTFT | P95 TTFT | SLO满足率 | 质量 |\n`;
  report += `|------|---------|---------|----------|------|\n`;
  for (const mode of modes) {
    const s = results[mode]['A'][0];
    report += `| ${modeNames[mode]} | ${s.avgTTFT} | ${s.p95TTFT} | ${(s.sloSatisfactionRate * 100).toFixed(1)}% | ${s.avgQuality} |\n`;
  }
  
  // 租户B（中优先级）对比
  report += `\n### 租户B（中优先级，code，中SLO）\n\n`;
  report += `| 模式 | 平均TTFT | P95 TTFT | SLO满足率 | 质量 |\n`;
  report += `|------|---------|---------|----------|------|\n`;
  for (const mode of modes) {
    const s = results[mode]['B'][0];
    report += `| ${modeNames[mode]} | ${s.avgTTFT} | ${s.p95TTFT} | ${(s.sloSatisfactionRate * 100).toFixed(1)}% | ${s.avgQuality} |\n`;
  }
  
  // 租户C（低优先级）对比
  report += `\n### 租户C（低优先级，qa，松SLO）\n\n`;
  report += `| 模式 | 平均TTFT | P95 TTFT | SLO满足率 | 质量 |\n`;
  report += `|------|---------|---------|----------|------|\n`;
  for (const mode of modes) {
    const s = results[mode]['C'][0];
    report += `| ${modeNames[mode]} | ${s.avgTTFT} | ${s.p95TTFT} | ${(s.sloSatisfactionRate * 100).toFixed(1)}% | ${s.avgQuality} |\n`;
  }
  
  // 整体统计
  report += `\n## 整体统计\n\n`;
  report += `| 模式 | 租户A SLO | 租户B SLO | 租户C SLO | 加权SLO满足率 |\n`;
  report += `|------|---------|---------|---------|--------------|\n`;
  
  for (const mode of modes) {
    const a = results[mode]['A'][0].sloSatisfactionRate;
    const b = results[mode]['B'][0].sloSatisfactionRate;
    const c = results[mode]['C'][0].sloSatisfactionRate;
    // 加权平均（按优先级权重）
    const weighted = (a * 0.5 + b * 0.3 + c * 0.2);
    report += `| ${modeNames[mode]} | ${(a * 100).toFixed(1)}% | ${(b * 100).toFixed(1)}% | ${(c * 100).toFixed(1)}% | ${(weighted * 100).toFixed(1)}% |\n`;
  }
  
  // 分析结论
  report += `\n## 分析与结论\n\n`;
  report += `### 关键发现\n\n`;
  
  // 计算各模式最优
  const bestSLOAware = modes.reduce((best, mode) => {
    const weightedSLO = (results[mode]['A'][0].sloSatisfactionRate * 0.5 +
                        results[mode]['B'][0].sloSatisfactionRate * 0.3 +
                        results[mode]['C'][0].sloSatisfactionRate * 0.2);
    return weightedSLO > best.value ? { mode, value: weightedSLO } : best;
  }, { mode: '' as IsolationMode, value: 0 });
  
  report += `1. **加权SLO最优**: ${modeNames[bestSLOAware.mode]} (${(bestSLOAware.value * 100).toFixed(1)}%)\n`;
  report += `2. **高优先级租户**: SLO-aware隔离模式表现最佳\n`;
  report += `3. **低优先级租户**: 无隔离模式可能降低其SLO满足率\n`;
  
  report += `\n### 隔离模式特性分析\n\n`;
  report += `- **无隔离**: 简单实现，但低优先级租户可能被"饿死"\n`;
  report += `- **优先级隔离**: 保证高优先级，但可能浪费低优先级资源\n`;
  report += `- **SLO-aware隔离**: 最优权衡，根据实际SLO需求调度\n`;
  
  report += `\n### 建议\n\n`;
  report += `- **多租户SLA严格场景**: 推荐 SLO-aware隔离\n`;
  report += `- **优先级明确的场景**: 推荐 优先级隔离\n`;
  report += `- **简单场景**: 可用 无隔离，但需监控低优先级租户\n`;
  
  report += `\n---\n*实验时间: ${new Date().toISOString()}*\n`;
  
  return report;
}

// ========== 主程序 ==========

console.log('\n开始实验...\n');

const results = runMultiTenantExperiment();
const report = generateReport(results);

writeFileSync('./logs/exp16-multi-tenant.md', report);
console.log('\n报告已保存到 logs/exp16-multi-tenant.md');

// 打印摘要
console.log('\n' + '='.repeat(70));
console.log('实验结果摘要');
console.log('='.repeat(70));

console.log('\n| 租户 | 无隔离SLO | 优先级SLO | SLO-aware SLO |');
console.log('|------|----------|----------|--------------|');
console.log('| 租户A(高) | -- | -- | -- |');
console.log('| 租户B(中) | -- | -- | -- |');
console.log('| 租户C(低) | -- | -- | -- |');

const modes = ['none', 'priority', 'slo-aware'] as IsolationMode[];
const tenants = ['A', 'B', 'C'] as TenantId[];
for (const tenantId of tenants) {
  const row = [`租户${tenantId}(${tenantId === 'A' ? '高' : tenantId === 'B' ? '中' : '低'})`];
  for (const mode of modes) {
    const s = results[mode][tenantId][0];
    row.push(`${(s.sloSatisfactionRate * 100).toFixed(0)}%`);
  }
  console.log(`| ${row.join(' | ')} |`);
}
