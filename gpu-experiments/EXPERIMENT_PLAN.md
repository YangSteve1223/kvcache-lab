# kvcache-lab GPU实验方案

**项目**: Runtime KV Memory Management for PD-Disaggregated LLM Serving  
**日期**: 2026-05-27  
**状态**: 待团队确认后执行

---

## 一、实验环境

| 项目 | 配置 |
|------|------|
| GPU | 1× RTX PRO 6000 (96GB VRAM) |
| 运行方式 | 单机双进程（P进程 + D进程，通过localhost TCP通信） |
| 带宽模拟 | Linux tc限速（100 Gbps / 25 Gbps / 无限速 三档） |
| 模型 | Qwen1.5-7B-Chat（优先）/ Llama-2-7B-Chat |
| 推理框架 | vLLM + transformers（底层API实现PD分离） |
| Python | 3.10+ |
| CUDA | 12.x |

### 显存预算（7B FP16 + 32K上下文）

| 组件 | 显存占用 |
|------|---------|
| 模型权重 | ~14 GB |
| KV Cache (32K) | ~8 GB |
| 激活值+临时buffer | ~4 GB |
| **单进程合计** | **~26 GB** |
| **双进程合计** | **~52 GB（96GB余量充足）** |

---

## 二、带宽档位设置

| 档位 | 带宽 | 对应真实场景 | tc命令 |
|------|------|------------|--------|
| **标准** | 12.5 GB/s (100 Gbps) | RDMA/RoCE跨节点（最常见） | `sudo tc qdisc add dev lo root netem rate 12.5gbit` |
| **拥塞** | 3.125 GB/s (25 Gbps) | 网络拥塞/低配集群 | `sudo tc qdisc add dev lo root netem rate 3.125gbit` |
| **理想** | localhost不限速 | 同机NVLink上界参考 | 无需tc |

---

## 三、实验设计（6组实验）

### Experiment G1: 单卡基线推理

**目的**: 建立无PD分离的性能基线

| 参数 | 值 |
|------|-----|
| 模型 | Qwen1.5-7B-Chat |
| 上下文长度 | 1K / 4K / 8K / 16K / 32K |
| 生成长度 | 128 tokens |
| 重复次数 | 3次取平均 |

**指标**: TTFT, TPOT, Throughput(tokens/s), GPU Memory Peak, Perplexity

---

### Experiment G2: PD分离基线

**目的**: 验证PD分离本身的开销和正确性

| 参数 | 值 |
|------|-----|
| 模型 | Qwen1.5-7B-Chat |
| 上下文长度 | 4K / 8K / 16K / 32K |
| 带宽 | 12.5 GB/s / 3.125 GB/s / 无限速 |
| KV传输方式 | 全量传输（100% KV） |
| 生成长度 | 128 tokens |
| 重复次数 | 3次 |

**指标**: TTFT(含传输), KV传输时间, KV数据量, TPOT, Perplexity

**预期**: 12.5GB/s下传输32K KV约41ms，占TTFT比例5-15%

---

### Experiment G3: Transmission-Aware Attention（核心创新验证）

**目的**: 验证TAA在真实GPU上的效果——这是论文的核心mechanism

#### G3a: α参数扫描

| 参数 | 值 |
|------|-----|
| α值 | 0 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 |
| 上下文长度 | 8K / 32K |
| 带宽 | 3.125 GB/s（拥塞场景，TAA价值最大） |
| 生成长度 | 128 tokens |

**核心公式**: `score_i = relevance_i + α × (-cost_normalized_i)`

#### G3b: 带宽×TAA矩阵

| 参数 | 值 |
|------|-----|
| 带宽 | 3.125 GB/s / 12.5 GB/s / 无限速 |
| α值 | 0(基线) / 0.1 / 0.3 |
| 上下文长度 | 8K / 32K |

#### G3c: TAA质量影响

| 参数 | 值 |
|------|-----|
| α值 | 0 / 0.05 / 0.1 / 0.2 / 0.3 / 0.5 |
| 生成长度 | 512 tokens（更长生成，质量差异更明显） |
| 评估 | Perplexity + 人工抽样检查生成质量 |

**指标**: TTFT, TPOT, Perplexity, Attention分布偏移度, α=0 vs α>0的延迟改善%

**预期结果**:
- α=0.1-0.3: 延迟改善5-15%，Perplexity上升<2%
- 高拥塞(3.125GB/s): 改善最大
- 低拥塞(12.5GB/s): 改善较小但仍有

**失败判据**: Perplexity上升>5%或延迟无改善 → 降级为"complementary technique"

---

### Experiment G4: Semantic Working Set

**目的**: 验证只传部分KV能否保持质量

| 参数 | 值 |
|------|-----|
| Working Set比例 | 30% / 50% / 70% / 100%(基线) |
| 上下文长度 | 4K / 8K / 16K / 32K |
| 带宽 | 12.5 GB/s |
| Working Set选择方式 | 基于attention density（最近Δ=32步内attention频率>θ） |
| 生成长度 | 128 tokens |
| 重复次数 | 3次 |

**指标**: KV传输量(MB), TTFT, Perplexity, 质量损失率

**预期结果**:
- 50% SWS: 传输量减半，Perplexity损失<3%
- 30% SWS: 传输量降70%，Perplexity损失5-10%（可接受范围取决于场景）
- 长上下文(32K): SWS收益最大（绝对传输量节省最多）

**失败判据**: 50% SWS下Perplexity损失>10% → 需要调整选择算法

---

### Experiment G5: Predictive Eviction

**目的**: 验证Predictive Eviction优于LRU

| 参数 | 值 |
|------|-----|
| GPU KV Cache限制 | 正常容量的 50% / 70% / 90% |
| 驱逐策略 | LRU / LFU / Predictive (Belady-inspired, 32-step horizon) |
| 上下文长度 | 8K / 32K |
| 生成长度 | 512 tokens（长生成，eviction效果更明显） |
| 重复次数 | 3次 |

**指标**: Cache命中率, 驱逐次数, 重加载延迟, Perplexity

**预期结果**:
- Predictive命中率比LRU高10-20%
- 重加载次数显著减少
- Perplexity损失更小

**失败判据**: Predictive与LRU差异<5% → 说明当前预测算法不够好，需改进或降级为"alternative policy"

---

### Experiment G6: Full Runtime OS集成

**目的**: 验证TAA + SWS + Predictive Eviction联合效果

| 参数 | 值 |
|------|-----|
| 配置 | Full OS / TAA-only / SWS-only / 无优化(基线) |
| 上下文长度 | 8K / 32K |
| 带宽 | 3.125 GB/s / 12.5 GB/s |
| 生成长度 | 128 / 512 tokens |
| 重复次数 | 3次 |

**指标**: TTFT, TPOT, 总传输量, Perplexity, 综合评分

**预期结果**:
- Full OS vs 基线: TTFT改善15-25%, 传输量减少40-60%
- 单独策略: TAA改善延迟，SWS改善传输量，Eviction改善内存效率
- 联合效果 > 任意单独策略

---

## 四、评估指标汇总

| 指标 | 含义 | 测量方式 |
|------|------|---------|
| TTFT | Time To First Token | 从请求到第一个token的时间 |
| TPOT | Time Per Output Token | 每个生成token的平均时间 |
| Throughput | 吞吐量 | tokens/s |
| Perplexity | 生成质量 | 在固定prompts上计算 |
| KV Transfer Size | KV传输量 | MB |
| KV Transfer Time | KV传输延迟 | ms |
| Cache Hit Rate | 缓存命中率 | % |
| GPU Memory Peak | 峰值显存占用 | GB |
| SLO Satisfaction | SLO满足率 | TTFT<阈值的请求比例 |

---

## 五、实验时间预估

| 实验 | 预估时间 | 说明 |
|------|---------|------|
| G1 基线 | 30 min | 5个上下文长度 × 3次 |
| G2 PD分离基线 | 45 min | 4长度 × 3带宽 × 3次 |
| G3 TAA验证 | 60 min | 参数扫描 + 带宽矩阵 + 质量评估 |
| G4 SWS验证 | 45 min | 4比例 × 4长度 × 3次 |
| G5 Eviction验证 | 45 min | 3容量 × 3策略 × 3次 |
| G6 Full OS集成 | 30 min | 4配置 × 2长度 × 2带宽 × 3次 |
| **总计** | **~4小时** | 含环境配置约5小时 |

---

## 六、费用预估

| 项目 | 费用 |
|------|------|
| RTX PRO 6000 96GB × 1台 | ~15-25元/小时 |
| 实验时间 5小时 | 75-125元 |
| **总计** | **约100元** |

---

## 七、风险预案

| 风险 | 概率 | 应对 |
|------|------|------|
| TAA效果弱(延迟改善<5%) | 30% | 降低α，改为adaptive α；TAA降级为supporting technique |
| Perplexity损失大(>5%) | 20% | 降低α到0.05，加quality guard |
| SWS质量损失大 | 15% | 提高Working Set比例到70%，或改进选择算法 |
| OOM | 10% | 降低上下文长度上限到16K |
| vLLM兼容性问题 | 20% | 用transformers底层API手写PD分离 |
| 模型下载慢 | 30% | 使用AutoDL镜像/模型缓存 |

---

## 八、论文用途规划

| 实验 | 论文章节 | 核心结论支撑 |
|------|---------|------------|
| G1 | §5.1 Baseline | 建立基线数字 |
| G2 | §5.2 PD Separation | PD分离开销分析 |
| G3 | §5.3 TAA Evaluation | **核心创新验证** |
| G4 | §5.4 SWS Evaluation | Runtime Policy验证 |
| G5 | §5.5 Eviction Evaluation | Runtime Policy验证 |
| G6 | §5.6 End-to-End | 系统整体效果 |

---

## 九、确认清单

- [ ] 团队确认实验环境和模型选择
- [ ] 确认带宽档位设置合理
- [ ] 确认α参数范围（0-0.5）
- [ ] 确认Working Set比例范围（30%-100%）
- [ ] 确认Perplexity可接受损失阈值（建议≤3%）
- [ ] 确认GPU实例已租用，SSH可访问
