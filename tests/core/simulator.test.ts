/**
 * PDSimulator 核心仿真器测试
 * 
 * 测试PD分离模拟器的各项功能
 */

import { describe, it, expect, beforeEach } from 'node:test';
import assert from 'node:assert';
import { PDSimulator, DEFAULT_CONFIG } from '../../src/core/PDSimulator.ts';
import { ServingRequest, SimulatorConfig, TaskType } from '../../src/core/types.ts';

/**
 * 生成测试请求
 */
function createTestRequest(overrides: Partial<ServingRequest> = {}): ServingRequest {
  return {
    id: `req-${Math.random().toString(36).substr(2, 9)}`,
    inputTokens: 1000,
    outputTokens: 100,
    taskType: 'math',
    arrivalTimeMs: Date.now(),
    ...overrides
  };
}

describe('PDSimulator', () => {
  let simulator: PDSimulator;

  beforeEach(() => {
    simulator = new PDSimulator();
  });

  // ========== 基础功能测试 ==========

  it('应该使用默认配置初始化', () => {
    const config = simulator.getConfig();
    expect(config.prefillBaseMs).toBe(DEFAULT_CONFIG.prefillBaseMs);
    expect(config.prefillMsPerToken).toBe(DEFAULT_CONFIG.prefillMsPerToken);
    expect(config.decodeBaseMs).toBe(DEFAULT_CONFIG.decodeBaseMs);
    expect(config.decodeMsPerToken).toBe(DEFAULT_CONFIG.decodeMsPerToken);
    expect(config.kvBytesPerToken).toBe(DEFAULT_CONFIG.kvBytesPerToken);
  });

  it('应该允许自定义配置', () => {
    const customConfig: Partial<SimulatorConfig> = {
      prefillBaseMs: 100,
      bandwidthBytesPerMs: 2000
    };
    const customSimulator = new PDSimulator(customConfig);
    const config = customSimulator.getConfig();
    expect(config.prefillBaseMs).toBe(100);
    expect(config.bandwidthBytesPerMs).toBe(2000);
  });

  it('应该正确重置模拟器状态', () => {
    const request = createTestRequest();
    simulator.simulateRequest(request);
    expect(simulator.getResults().length).toBe(1);
    
    simulator.reset();
    expect(simulator.getResults().length).toBe(0);
  });

  it('应该允许设置随机种子', () => {
    simulator.setSeed(123);
    // 重置后再次设置应该工作
    simulator.setSeed(456);
    expect(true).toBe(true); // 不抛异常即通过
  });

  // ========== 请求处理测试 ==========

  it('应该模拟单个请求并返回结果', () => {
    const request = createTestRequest({
      inputTokens: 500,
      outputTokens: 50
    });
    
    const result = simulator.simulateRequest(request);
    
    expect(result.requestId).toBe(request.id);
    expect(result.ttftMs).toBeGreaterThan(0);
    expect(result.e2eLatencyMs).toBeGreaterThan(result.ttftMs);
    expect(result.taskType).toBe(request.taskType);
  });

  it('应该正确处理缓存命中', () => {
    const prefixHash = 'test-prefix-hash';
    
    // 第一次请求，缓存未命中
    const request1 = createTestRequest({ prefixHash });
    const result1 = simulator.simulateRequest(request1);
    expect(result1.cacheHit).toBe(false);
    
    // 第二次相同前缀请求，缓存命中
    const request2 = createTestRequest({ prefixHash, id: 'req-2' });
    const result2 = simulator.simulateRequest(request2);
    expect(result2.cacheHit).toBe(true);
  });

  it('应该正确处理不同任务类型', () => {
    const taskTypes: TaskType[] = ['math', 'code', 'qa', 'conversation'];
    
    for (const taskType of taskTypes) {
      const request = createTestRequest({ taskType });
      const result = simulator.simulateRequest(request);
      expect(result.taskType).toBe(taskType);
    }
  });

  it('应该正确处理不同batch大小', () => {
    const request = createTestRequest();
    
    // batch=1
    const result1 = simulator.simulateRequest(request, null, 1);
    
    // batch=8
    simulator.reset();
    const result8 = simulator.simulateRequest(request, null, 8);
    
    // batch越大，延迟应该越高
    expect(result8.ttftMs).toBeGreaterThanOrEqual(result1.ttftMs);
  });

  // ========== 批量处理测试 ==========

  it('应该模拟批量请求', () => {
    const requests = Array.from({ length: 10 }, (_, i) => 
      createTestRequest({ id: `batch-req-${i}` })
    );
    
    const stats = simulator.simulateBatch(requests);
    
    expect(stats.totalRequests).toBe(10);
    expect(stats.avgTTFT).toBeGreaterThan(0);
    expect(stats.avgE2E).toBeGreaterThan(0);
  });

  it('应该正确计算TTFT百分位数', () => {
    const requests = Array.from({ length: 100 }, (_, i) => 
      createTestRequest({ 
        id: `percentile-req-${i}`,
        inputTokens: 500 + Math.floor(Math.random() * 1000)
      })
    );
    
    const stats = simulator.simulateBatch(requests);
    
    expect(stats.p50TTFT).toBeGreaterThan(0);
    expect(stats.p95TTFT).toBeGreaterThanOrEqual(stats.p50TTFT);
    expect(stats.p99TTFT).toBeGreaterThanOrEqual(stats.p95TTFT);
  });

  it('应该正确按任务类型分组统计', () => {
    const requests = [
      createTestRequest({ taskType: 'math' }),
      createTestRequest({ taskType: 'math' }),
      createTestRequest({ taskType: 'code' }),
      createTestRequest({ taskType: 'qa' })
    ];
    
    const stats = simulator.simulateBatch(requests);
    
    expect(stats.perTaskStats.math.count).toBe(2);
    expect(stats.perTaskStats.code.count).toBe(1);
    expect(stats.perTaskStats.qa.count).toBe(1);
  });

  // ========== 压缩配置测试 ==========

  it('应该处理无压缩配置', () => {
    const request = createTestRequest();
    const result = simulator.simulateRequest(request, null);
    
    expect(result.compressionRatio).toBe(1.0);
    expect(result.qualityScore).toBe(1.0);
  });

  it('应该处理不同的压缩保留率', () => {
    const request = createTestRequest();
    
    // 使用自定义压缩配置
    const compressionConfig = {
      strategy: 'uniform' as const,
      pLayers: Array.from({ length: 32 }, (_, i) => ({
        layerIndex: i,
        totalLayers: 32,
        retentionRatio: 0.5,
        keyPrecision: 16,
        valuePrecision: 8
      })),
      dLayers: []
    };
    
    const result = simulator.simulateRequest(request, compressionConfig);
    
    expect(result.compressionRatio).toBeLessThan(1.0);
    expect(result.qualityScore).toBeLessThan(1.0);
  });

  // ========== 缓存管理器测试 ==========

  it('应该正确获取缓存管理器', () => {
    const cacheManager = simulator.getCacheManager();
    expect(cacheManager).toBeDefined();
    expect(typeof cacheManager.lookup).toBe('function');
    expect(typeof cacheManager.store).toBe('function');
  });

  it('应该正确计算缓存命中率', () => {
    const prefixHash = 'hit-test-prefix';
    
    // 发送10个请求，5个相同前缀
    for (let i = 0; i < 5; i++) {
      simulator.simulateRequest(createTestRequest({ prefixHash, id: `hit-${i}` }));
    }
    for (let i = 0; i < 5; i++) {
      simulator.simulateRequest(createTestRequest({ prefixHash: `other-${i}`, id: `miss-${i}` }));
    }
    
    const stats = simulator.computeStats();
    // 5个命中/10个总数 = 0.5
    expect(stats.cacheHitRate).toBeGreaterThanOrEqual(0.4);
    expect(stats.cacheHitRate).toBeLessThanOrEqual(0.6);
  });

  // ========== 吞吐量测试 ==========

  it('应该正确计算吞吐量', () => {
    const requests = Array.from({ length: 20 }, (_, i) => 
      createTestRequest({ id: `throughput-req-${i}` })
    );
    
    const stats = simulator.simulateBatch(requests);
    
    expect(stats.throughputTokensPerSec).toBeGreaterThan(0);
  });

  // ========== 边界条件测试 ==========

  it('应该处理空请求列表', () => {
    const stats = simulator.simulateBatch([]);
    
    expect(stats.totalRequests).toBe(0);
    expect(stats.avgTTFT).toBe(0);
    expect(stats.avgE2E).toBe(0);
  });

  it('应该处理超大token数量', () => {
    const request = createTestRequest({
      inputTokens: 100000,
      outputTokens: 10000
    });
    
    const result = simulator.simulateRequest(request);
    
    expect(result.ttftMs).toBeGreaterThan(0);
    expect(result.e2eLatencyMs).toBeGreaterThan(result.ttftMs);
  });

  it('应该处理零token数量', () => {
    const request = createTestRequest({
      inputTokens: 0,
      outputTokens: 0
    });
    
    const result = simulator.simulateRequest(request);
    
    expect(result.ttftMs).toBeGreaterThanOrEqual(0);
    expect(result.e2eLatencyMs).toBeGreaterThanOrEqual(0);
  });
});
