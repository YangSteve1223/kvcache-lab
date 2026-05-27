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
| 带宽模拟 | Linux tc限速（100 Gbps / 25 Gbps / 12.5 Gbps / 无限速 四档） |
| 主实验模型 | Qwen2.5-7B-Instruct |
| Scaling模型 | Qwen2.5-14B-Instruct（G1/G2/G3核心实验） |
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
| **标准** | 12.5 GB/s (100 Gbps) | RDMA/RoCE跨节点（最常见） | `sudo tc qdisc add dev lo root netem rate 100gbit` |
| **拥塞** | 3.125 GB/s (25 Gbps) | 网络拥塞/低配集群 | `sudo tc qdisc add dev lo root netem rate 25gbit` |
| **低配** | 1.56 GB/s (12.5 Gbps) | 低配以太网 | `sudo tc qdisc add dev lo root netem rate 12.5gbit` |
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
| 重复次数 | 5次取平均（latency指标需≥10次） |

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
| 重复次数 | 5次（latency 10次） |

**指标**: TTFT(含传输), KV传输时间, KV数据量, TPOT, Perplexity

**预期**: 12.5GB/s下传输32K KV约41ms，占TTFT比例5-15%

---

### Experiment G3: Transmission-Aware Attention（核心创新验证）

**目的**: 验证TAA在真实GPU上的效果——这是论文的核心mechanism

#### G3a: α参数扫描

| 参数 | 值 |
|------|-----|
| α值 | 0 / 0.01 / 0.03 / 0.05 / 0.1 / 0.15 / 0.2 |
| 上下文长度 | 8K / 32K |
| 带宽 | 3.125 GB/s（拥塞场景，TAA价值最大） |
| 生成长度 | 128 tokens |

**核心公式**: `b_i = -α × tanh((cost_i - μ) / σ)`, `score_i = relevance_i + b_i`

tanh确保bounded在[-1,1]，比纯z-score更安全，避免极端cost导致attention collapse。
σ=0时退化为普通attention（无偏置）。

#### G3b: 带宽×TAA矩阵

| 参数 | 值 |
|------|-----|
| 带宽 | 3.125 GB/s / 12.5 GB/s / 无限速 |
| α值 | 0(基线) / 0.1 / 0.3 |
| 上下文长度 | 8K / 32K |

#### G3c: TAA质量影响

| 参数 | 值 |
|------|-----|
| α值 | 0 / 0.01 / 0.03 / 0.05 / 0.1 / 0.15 / 0.2 |
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
| 重复次数 | 5次（latency 10次） |

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
| 驱逐策略 | LRU / LFU / Predictive (Belady-inspired) |
| 预测horizon | 8 / 16 / 32 / 64 steps（sensitivity analysis） |
| 上下文长度 | 8K / 32K |
| 生成长度 | 512 tokens（长生成，eviction效果更明显） |
| 重复次数 | 5次（latency 10次） |

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
| 配置 | baseline / TAA-only / SWS-only / Eviction-only / TAA+SWS / TAA+Eviction / SWS+Eviction / Full OS |
| 上下文长度 | 8K / 32K |
| 带宽 | 3.125 GB/s / 12.5 GB/s |
| 生成长度 | 128 / 512 tokens |
| 重复次数 | 5次（latency 10次） |

**指标**: TTFT, TPOT, 总传输量, Perplexity, 综合评分

**预期结果**:
- Full OS vs 基线: TTFT改善15-25%, 传输量减少40-60%
- 单独策略: TAA改善延迟，SWS改善传输量，Eviction改善内存效率
- 联合效果 > 任意单独策略

---

### Experiment G7: Multi-Request Concurrent Serving

**目的**: 验证TAA在真实serving场景（多请求竞争+拥塞）下的效果——这是TAA最大价值的场景

| 参数 | 值 |
|------|-----|
| 并发请求数 | 8 / 16 / 32 |
| 请求到达模式 | Poisson (λ按并发数调整) |
| 请求长度分布 | 混合(ShareGPT-style): 短(1K)/中(4K)/长(16K) |
| 带宽 | 3.125 GB/s / 12.5 GB/s |
| 配置 | baseline / TAA / Full OS |
| 总请求数 | ≥500 per config |
| warmup | 50 requests |
| measurement | 后450 requests |

**指标**: TTFT P50/P95/P99, TPOT P50/P95, Throughput(tokens/s), SLO violation rate(TTFT>1s), Queue waiting time

**预期结果**:
- TAA在8+并发时TTFT P95改善15-30%
- 拥塞(3.125GB/s)下TAA SLO violation率显著降低
- 无限速时TAA与baseline差异小（证明TAA只在拥塞时有价值）

**失败判据**: TAA在32并发下SLO violation无改善 → TAA对real serving无效

---

### Experiment G8: Generation Quality Beyond Perplexity

**目的**: 补充任务级质量评估，不只看PPL

| 参数 | 值 |
|------|-----|
| 评估任务 | MT-Bench(chat) / LongBench(long-context) / Needle-in-Haystack(retrieval) |
| α值 | 0(baseline) / 0.05 / 0.1 / 0.2 |
| 样本量 | 每任务50-100条 |

**指标**: MT-Bench score, LongBench子任务accuracy, Needle accuracy, ROUGE-L

---

### Experiment G3d: TAA Layer Sensitivity

**目的**: 确定TAA应该在哪几层开启——全层开启可能导致semantic drift

| 配置 | 说明 |
|------|------|
| all layers | 所有32层都加TAA bias |
| last 1/3 | 后1/3层(layers 21-31)加TAA |
| last 1/2 | 后1/2层(layers 16-31)加TAA |
| decode-only | 只在decode阶段的attention加TAA |

| 固定参数 | 值 |
|---------|-----|
| α | 0.1 |
| 带宽 | 3.125 GB/s |
| 上下文 | 32K |
| 生成长度 | 512 tokens |

**指标**: TTFT, Perplexity, 生成质量(与baseline对比)

**预期**: last 1/3效果最好——early layers偏lexical不适合加bias，后层偏semantic更适合cost-aware reranking

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
| G6 Full OS集成 | 45 min | 8配置 × 2长度 × 2带宽 × 3次 |
| G7 多请求并发 | 60 min | 3并发 × 2带宽 × 3配置 × 60s |
| G8 生成质量评估 | 60 min | 3任务 × 4 α值 × 50-100条 |
| **总计** | **~7小时** | 含环境配置约8小时 |

---

## 五、GPU Profiling要求

TAA自身有计算开销，reviewer一定会问"overhead多少"。必须测量：

| 工具 | 测量内容 |
|------|---------|
| nvidia-smi dmon | GPU利用率/显存/拷贝带宽 |
| torch.profiler | TAA bias计算的额外时间 |
| 时间戳插桩 | z-score + tanh + bias injection各自的μs |

**TAA overhead预期**: <1% of attention time（只是一次tanh+加法）

---

## 六、实验环境声明

论文中必须明确写：

> "We use a single 96GB GPU with two processes connected via TCP over localhost, with Linux tc bandwidth throttling to emulate realistic network conditions. This is a single-node emulation of PD-disaggregated serving."

**不要写**: "real distributed cluster" 或 "multi-node deployment"

---

## 六、费用预估

| 项目 | 费用 |
|------|------|
| RTX PRO 6000 96GB × 1台 | ~15-25元/小时 |
| 实验时间 7小时 | 105-175元 |
| **总计** | **约140元** |

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
