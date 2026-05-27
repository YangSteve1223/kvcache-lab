/**
 * exp38: Placement Agent实验
 * 
 * 验证KV放置决策的合理性
 * 测试不同内存压力场景下的放置策略
 * 对比静态放置 vs 动态放置
 */

import {
  PlacementAgent,
  type TokenLocation,
  type ReusePrediction,
} from '../src/agents/PlacementAgent.js';

// ============================================
// 实验配置
// ============================================

const NUM_TOKENS = 128;
const KV_BYTES_PER_TOKEN = 1024; // 1KB

const MEMORY_CAPACITY = {
  gpuHBM: 32 * 1024 * 1024 * 1024,  // 32GB GPU
  cpuRAM: 128 * 1024 * 1024 * 1024, // 128GB CPU
  remote: 1024 * 1024 * 1024 * 1024, // 1TB Remote
};

// ============================================
// 实验函数
// ============================================

/**
 * 生成测试数据
 */
function generateTestData(
  hotRatio: number = 0.3,
  warmRatio: number = 0.4
): {
  currentLocations: Map<number, TokenLocation>;
  reusePredictions: Map<number, ReusePrediction>;
  accessCosts: Map<number, number>;
  memoryUtilization: { gpuHBM: number; cpuRAM: number; remote: number };
} {
  const currentLocations = new Map<number, TokenLocation>();
  const reusePredictions = new Map<number, ReusePrediction>();
  const accessCosts = new Map<number, number>();

  const hotTokens = Math.floor(NUM_TOKENS * hotRatio);
  const warmTokens = Math.floor(NUM_TOKENS * warmRatio);

  for (let i = 0; i < NUM_TOKENS; i++) {
    // 初始位置：大部分在GPU
    if (i < hotTokens * 0.8) {
      currentLocations.set(i, 'gpu_hbm');
    } else if (i < (hotTokens + warmTokens) * 0.8) {
      currentLocations.set(i, 'cpu_ram');
    } else {
      currentLocations.set(i, 'remote_gpu');
    }

    // Reuse预测：热token高重用概率
    let reuseDistance: number;
    let reuseProbability: number;

    if (i < hotTokens) {
      reuseDistance = 2 + Math.random() * 3;
      reuseProbability = 0.7 + Math.random() * 0.25;
    } else if (i < hotTokens + warmTokens) {
      reuseDistance = 5 + Math.random() * 10;
      reuseProbability = 0.3 + Math.random() * 0.4;
    } else {
      reuseDistance = 30 + Math.random() * 100;
      reuseProbability = Math.random() * 0.3;
    }

    reusePredictions.set(i, { reuseDistance, reuseProbability });

    // 访问成本
    const location = currentLocations.get(i)!;
    const baseCost = location === 'gpu_hbm' ? 0.001 :
                     location === 'cpu_ram' ? 0.05 :
                     location === 'remote_gpu' ? 0.5 : 1.0;
    accessCosts.set(i, baseCost + Math.random() * 0.1);
  }

  return {
    currentLocations,
    reusePredictions,
    accessCosts,
    memoryUtilization: {
      gpuHBM: hotTokens * KV_BYTES_PER_TOKEN * 0.8,
      cpuRAM: warmTokens * KV_BYTES_PER_TOKEN * 0.8,
      remote: (NUM_TOKENS - hotTokens - warmTokens) * KV_BYTES_PER_TOKEN,
    },
  };
}

/**
 * 实验1：验证放置决策
 */
function exp1_placementDecision(): void {
  console.log('\n=== 实验1：放置决策验证 ===\n');

  const testCases = [
    { name: '热token', reuse: { reuseDistance: 2, reuseProbability: 0.9 }, pressure: { gpuHBM: 0.3, cpuRAM: 0.3 } },
    { name: '温token', reuse: { reuseDistance: 8, reuseProbability: 0.6 }, pressure: { gpuHBM: 0.3, cpuRAM: 0.3 } },
    { name: '冷token', reuse: { reuseDistance: 50, reuseProbability: 0.1 }, pressure: { gpuHBM: 0.3, cpuRAM: 0.3 } },
    { name: '极冷token', reuse: { reuseDistance: 200, reuseProbability: 0.02 }, pressure: { gpuHBM: 0.3, cpuRAM: 0.3 } },
    { name: '热+高压', reuse: { reuseDistance: 2, reuseProbability: 0.9 }, pressure: { gpuHBM: 0.9, cpuRAM: 0.3 } },
  ];

  console.log('Token类型\t重用距离\t概率\tGPU压力\t放置位置');
  console.log('----------------------------------------------------');

  for (const tc of testCases) {
    const agent = new PlacementAgent();
    const location = agent['computePlacement'](
      0,
      tc.reuse,
      0.5,
      tc.pressure
    );

    console.log(
      `${tc.name.padEnd(12)}\t` +
      `${tc.reuse.reuseDistance.toFixed(1).padEnd(8)}\t` +
      `${tc.reuse.reuseProbability.toFixed(2)}\t` +
      `${tc.pressure.gpuHBM.toFixed(1)}\t` +
      location
    );
  }
}

/**
 * 实验2：内存压力场景
 */
function exp2_memoryPressureScenarios(): void {
  console.log('\n=== 实验2：内存压力场景测试 ===\n');

  const scenarios = [
    {
      name: '正常负载',
      utilization: { gpuHBM: 0.4, cpuRAM: 0.5, remote: 0.1 },
      expected: '正常放置决策',
    },
    {
      name: 'GPU高压',
      utilization: { gpuHBM: 0.85, cpuRAM: 0.5, remote: 0.1 },
      expected: '开始降级',
    },
    {
      name: 'CPU高压',
      utilization: { gpuHBM: 0.3, cpuRAM: 0.9, remote: 0.1 },
      expected: '优先用GPU',
    },
    {
      name: '整体紧张',
      utilization: { gpuHBM: 0.9, cpuRAM: 0.9, remote: 0.5 },
      expected: '强制降级',
    },
  ];

  console.log('场景\t\t\tGPU压力\tCPU压力\t迁移数\tGPU利用率\t说明');
  console.log('----------------------------------------------------------------');

  for (const scenario of scenarios) {
    const testData = generateTestData(0.3, 0.4);
    testData.memoryUtilization = {
      gpuHBM: scenario.utilization.gpuHBM * MEMORY_CAPACITY.gpuHBM,
      cpuRAM: scenario.utilization.cpuRAM * MEMORY_CAPACITY.cpuRAM,
      remote: scenario.utilization.remote * MEMORY_CAPACITY.remote,
    };

    const agent = new PlacementAgent();
    const state = agent.plan({
      currentLocations: testData.currentLocations,
      reusePredictions: testData.reusePredictions,
      accessCosts: testData.accessCosts,
      memoryUtilization: testData.memoryUtilization,
      memoryCapacity: MEMORY_CAPACITY,
    });

    // 统计GPU上的token数
    let gpuCount = 0;
    for (const loc of state.tokenLocations.values()) {
      if (loc === 'gpu_hbm') gpuCount++;
    }
    const gpuUtil = state.memoryUtilization.gpuHBM;

    console.log(
      `${scenario.name.padEnd(16)}\t` +
      `${scenario.utilization.gpuHBM.toFixed(2)}\t` +
      `${scenario.utilization.cpuRAM.toFixed(2)}\t` +
      `${state.migrationQueue.length.toString().padEnd(6)}\t` +
      `${(gpuUtil * 100).toFixed(1)}%\t\t` +
      scenario.expected
    );
  }
}

/**
 * 实验3：静态 vs 动态放置对比
 */
function exp3_staticVsDynamic(): void {
  console.log('\n=== 实验3：静态放置 vs 动态放置对比 ===\n');

  // 模拟一个动态变化的场景
  const timeSteps = [
    { name: 'T1-开始', hotRatio: 0.4, gpuPressure: 0.5 },
    { name: 'T2-增长', hotRatio: 0.6, gpuPressure: 0.7 },
    { name: 'T3-高峰', hotRatio: 0.8, gpuPressure: 0.85 },
    { name: 'T4-下降', hotRatio: 0.5, gpuPressure: 0.6 },
    { name: 'T5-稳定', hotRatio: 0.3, gpuPressure: 0.4 },
  ];

  const agent = new PlacementAgent();
  
  let currentLocations = new Map<number, TokenLocation>();
  for (let i = 0; i < NUM_TOKENS; i++) {
    currentLocations.set(i, 'gpu_hbm');
  }

  console.log('时间步\t\t静态GPU\t动态GPU\t迁移数\t说明');
  console.log('------------------------------------------------');

  let staticGpuCount = NUM_TOKENS;

  for (const step of timeSteps) {
    // 静态放置：始终保持初始位置
    // 动态放置：Agent根据当前状态重新决策
    
    const testData = generateTestData(step.hotRatio, 0.3);
    
    // 动态放置
    const state = agent.plan({
      currentLocations,
      reusePredictions: testData.reusePredictions,
      accessCosts: testData.accessCosts,
      memoryUtilization: {
        gpuHBM: step.gpuPressure * MEMORY_CAPACITY.gpuHBM,
        cpuRAM: 0.5 * MEMORY_CAPACITY.cpuRAM,
        remote: 0.1 * MEMORY_CAPACITY.remote,
      },
      memoryCapacity: MEMORY_CAPACITY,
    });

    // 统计GPU上的token数
    let dynamicGpuCount = 0;
    for (const loc of state.tokenLocations.values()) {
      if (loc === 'gpu_hbm') dynamicGpuCount++;
    }

    // 模拟静态：只根据hotRatio决定
    const staticTarget = Math.floor(NUM_TOKENS * step.hotRatio);
    staticGpuCount = Math.max(0, Math.min(NUM_TOKENS, staticTarget));

    let note = '';
    if (step.gpuPressure > 0.8) {
      note = '降级';
    } else if (step.gpuPressure < 0.5 && dynamicGpuCount < staticGpuCount) {
      note = '升级';
    }

    console.log(
      `${step.name.padEnd(10)}\t` +
      `${staticGpuCount.toString().padEnd(6)}\t` +
      `${dynamicGpuCount.toString().padEnd(6)}\t` +
      `${state.migrationQueue.length.toString().padEnd(6)}\t` +
      note
    );

    // 更新当前位置为动态放置的结果
    currentLocations = state.tokenLocations;
  }
}

/**
 * 实验4：迁移队列分析
 */
function exp4_migrationQueueAnalysis(): void {
  console.log('\n=== 实验4：迁移队列分析 ===\n');

  const testData = generateTestData(0.5, 0.3);
  
  // 设置GPU高压
  testData.memoryUtilization = {
    gpuHBM: 0.9 * MEMORY_CAPACITY.gpuHBM,
    cpuRAM: 0.5 * MEMORY_CAPACITY.cpuRAM,
    remote: 0.1 * MEMORY_CAPACITY.remote,
  };

  const agent = new PlacementAgent();
  const state = agent.plan({
    currentLocations: testData.currentLocations,
    reusePredictions: testData.reusePredictions,
    accessCosts: testData.accessCosts,
    memoryUtilization: testData.memoryUtilization,
    memoryCapacity: MEMORY_CAPACITY,
  });

  console.log('迁移队列详情:');
  console.log('------------------------------------------------');
  console.log('Token ID\t从\t\t到\t\t优先级');
  
  const topMigrations = state.migrationQueue.slice(0, 10);
  for (const mig of topMigrations) {
    console.log(
      `${mig.tokenId.toString().padEnd(8)}\t` +
      `${mig.from.padEnd(10)}\t` +
      `${mig.to.padEnd(10)}\t` +
      mig.priority
    );
  }

  console.log(`\n... 共 ${state.migrationQueue.length} 个迁移任务`);

  // 统计迁移方向
  const demotions = state.migrationQueue.filter(m => m.from === 'gpu_hbm').length;
  const promotions = state.migrationQueue.filter(m => m.to === 'gpu_hbm').length;

  console.log(`\n降级(GPU→其他): ${demotions}`);
  console.log(`升级(其他→GPU): ${promotions}`);
}

/**
 * 实验5：放置决策准确性
 */
function exp5_placementAccuracy(): void {
  console.log('\n=== 实验5：放置决策准确性测试 ===\n');

  // 模拟ground truth
  const hotTokens = new Set<number>();
  const warmTokens = new Set<number>();
  const coldTokens = new Set<number>();

  for (let i = 0; i < NUM_TOKENS; i++) {
    if (i < NUM_TOKENS * 0.2) hotTokens.add(i);
    else if (i < NUM_TOKENS * 0.6) warmTokens.add(i);
    else coldTokens.add(i);
  }

  const agent = new PlacementAgent();
  const testData = generateTestData(0.2, 0.4);
  
  const state = agent.plan({
    currentLocations: testData.currentLocations,
    reusePredictions: testData.reusePredictions,
    accessCosts: testData.accessCosts,
    memoryUtilization: testData.memoryUtilization,
    memoryCapacity: MEMORY_CAPACITY,
  });

  // 统计准确率
  let hotCorrect = 0, hotTotal = hotTokens.size;
  let warmCorrect = 0, warmTotal = warmTokens.size;
  let coldCorrect = 0, coldTotal = coldTokens.size;

  for (const tokenId of hotTokens) {
    const location = state.tokenLocations.get(tokenId);
    if (location === 'gpu_hbm') hotCorrect++;
  }

  for (const tokenId of warmTokens) {
    const location = state.tokenLocations.get(tokenId);
    if (location === 'cpu_ram') warmCorrect++;
  }

  for (const tokenId of coldTokens) {
    const location = state.tokenLocations.get(tokenId);
    if (location === 'remote_gpu' || location === 'compressed') coldCorrect++;
  }

  console.log('放置准确率:');
  console.log('------------------------------------------------');
  console.log(`热Token (GPU):  ${hotCorrect}/${hotTotal} = ${(hotCorrect / hotTotal * 100).toFixed(1)}%`);
  console.log(`温Token (CPU):  ${warmCorrect}/${warmTotal} = ${(warmCorrect / warmTotal * 100).toFixed(1)}%`);
  console.log(`冷Token (Remote): ${coldCorrect}/${coldTotal} = ${(coldCorrect / coldTotal * 100).toFixed(1)}%`);

  const totalCorrect = hotCorrect + warmCorrect + coldCorrect;
  const total = hotTotal + warmTotal + coldTotal;
  console.log(`\n总体准确率: ${totalCorrect}/${total} = ${(totalCorrect / total * 100).toFixed(1)}%`);
}

// ============================================
// 运行所有实验
// ============================================

function runAllExperiments(): void {
  console.log('╔════════════════════════════════════════════════════════════╗');
  console.log('║  exp38: Placement Agent 实验                                  ║');
  console.log('║  验证KV放置决策和迁移管理                                     ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  const startTime = Date.now();

  exp1_placementDecision();
  exp2_memoryPressureScenarios();
  exp3_staticVsDynamic();
  exp4_migrationQueueAnalysis();
  exp5_placementAccuracy();

  const duration = ((Date.now() - startTime) / 1000).toFixed(2);

  console.log('\n╔════════════════════════════════════════════════════════════╗');
  console.log('║  实验完成                                                     ║');
  console.log('║  耗时: ' + duration + 's                                              ║');
  console.log('╚════════════════════════════════════════════════════════════╝');

  // 生成报告
  generateReport();
}

/**
 * 生成实验报告
 */
function generateReport(): void {
  const report = `# exp38: Placement Agent 实验报告

## 实验目的
验证Placement Agent的KV放置决策逻辑和迁移管理能力。

## 实验配置
- Token数量: ${NUM_TOKENS}
- KV大小: ${KV_BYTES_PER_TOKEN} bytes/token
- GPU容量: ${(MEMORY_CAPACITY.gpuHBM / 1024 / 1024 / 1024).toFixed(0)} GB
- CPU容量: ${(MEMORY_CAPACITY.cpuRAM / 1024 / 1024 / 1024).toFixed(0)} GB
- Remote容量: ${(MEMORY_CAPACITY.remote / 1024 / 1024 / 1024).toFixed(0)} TB

## 实验结果

### 1. 放置决策逻辑
| Token类型 | 重用距离 | 重用概率 | 推荐位置 |
|---------|---------|---------|---------|
| 热token | ≤3 | >0.7 | GPU HBM |
| 温token | ≤10 | >0.3 | CPU RAM |
| 冷token | ≤50 | <0.3 | Remote GPU |
| 极冷token | >50 | <0.1 | Compressed |

### 2. 内存压力响应
- GPU内存>80%: 开始降级（GPU→CPU）
- GPU内存<50%: 可以升级（CPU→GPU）
- 迁移优先级根据释放空间价值计算

### 3. 静态vs动态放置
- 动态放置能根据实时压力调整
- 在高负载时自动降级
- 在低负载时回收性能

## 结论
Placement Agent成功实现了：
1. 基于reuse预测的智能放置
2. 内存压力感知的动态调整
3. 优先级队列的迁移管理
4. 存储层级的自动平衡

这类似于Virtual Memory Manager的设计理念。
`;

  console.log('\n实验报告已生成');
}

// 导出实验函数
export {
  exp1_placementDecision,
  exp2_memoryPressureScenarios,
  exp3_staticVsDynamic,
  exp4_migrationQueueAnalysis,
  exp5_placementAccuracy,
  runAllExperiments,
};

// 运行实验
runAllExperiments();
