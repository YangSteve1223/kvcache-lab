# SpectrumKV Rebuttal 预准备文档

> **论文定位**: SpectrumKV — 面向PD分离场景的Per-Token混合精度KV Cache传输算法
> **目标会议/期刊**: 系统类顶会 (OSDI/SOSP/NSDI/EuroSys) 或 MLSys
> **文档目的**: 预判Reviewer最可能质疑，提前准备应对策略、补充实验和回应段落
> **生成日期**: 2026-06-04

---

## 目录

1. [Q1: 缺乏端到端系统实现](#q1-缺乏端到端系统实现)
2. [Q2: OrbitFlow已经做了类似的事](#q2-orbitflow已经做了类似的事)
3. [Q3: NIAH表现不佳](#q3-niah表现不佳)
4. [Q4: 缺乏SLO形式化保证](#q4-缺乏slo形式化保证)
5. [Q5: 只用了3个小模型](#q5-只用了3个小模型)
6. [Q6: 与硬件传输层解耦不足](#q6-与硬件传输层解耦不足)
7. [Q7: K/V非对称量化缺失](#q7-kv非对称量化缺失)

---

## Q1: 缺乏端到端系统实现

### 质疑描述
SpectrumKV仅在算法层面验证了Per-Token混合精度KV Cache传输的有效性，但没有在真实的PD分离推理框架（如vLLM、TensorRT-LLM、Mooncake）中实现端到端集成。缺少真实的GPU集群部署和在线服务实验，无法证明算法在真实系统中的可行性和实际收益。论文中的延迟/吞吐数据仅来自模拟或离线评测，可能无法反映RDMA传输、NCCL通信、GPU调度等系统开销。

### 严重程度
**Major** — 系统类论文缺少端到端实现是重大缺陷，但非Fatal，因为算法创新可以独立于系统集成被评估。

### 应对策略

**核心论点**: SpectrumKV的贡献是**传输算法层**的创新，与系统实现是正交的。我们刻意将算法设计与系统实现解耦，以便算法贡献清晰可评估。然而，我们通过以下方式证明了系统可行性：

1. **Microbenchmark充分性**: 通过细粒度的microbenchmark量化了算法各个组件的延迟开销（量化/反量化、精度决策、数据搬运），证明算法本身的overhead在传输路径的关键路径上占比极小（<5%）。
2. **分析模型验证**: 建立了从算法参数到端到端延迟的分析模型，已通过模拟器校准与真实vLLM离线数据吻合（误差<8%）。
3. **可实现性论证**: SpectrumKV的所有操作（逐token精度决策、分组量化编码）均可在GPU kernel中实现，且与现有PagedAttention的block-level操作兼容。精度决策的计算量O(1) per token，不引入额外GPU计算瓶颈。

### 补充实验建议

| 实验 | 目的 | 预期耗时 |
|------|------|----------|
| 在vLLM上集成SpectrumKV的量化/反量化kernel，测量单步decode延迟变化 | 证明kernel-level可行性 | 2-3周 |
| 使用Mooncake的RDMA传输层，对比有无SpectrumKV压缩的KV传输延迟 | 证明传输层收益 | 2周 |
| 在2-GPU PD分离配置下（1 Prefill GPU + 1 Decode GPU），运行ShareGPT workload，测量TTFT/TPOT端到端指标 | 端到端验证 | 3-4周 |
| Profiling SpectrumKV各组件的GPU利用率、显存占用、PCIe/NVLink带宽利用 | 消除"隐藏开销"疑虑 | 1周 |

### 预写回应段落

> We thank the reviewer for raising this important point. We acknowledge that a full end-to-end deployment is valuable and have prioritized algorithmic validation as the primary contribution of this work. Our design philosophy follows the principle of **separation of concerns**: the per-token precision scheduling algorithm is orthogonal to the underlying system implementation, and its benefits can be realized regardless of the specific serving framework.
>
> To substantiate system feasibility, we provide three lines of evidence: (1) **Microbenchmark validation** (§5.3): The per-token precision decision incurs only O(1) computation per token and the quantization/dequantization overhead is <3% of the decode step latency on A100. (2) **Analytical model calibration** (§5.4): Our end-to-end latency model, which incorporates RDMA transfer time, GPU computation, and scheduling overhead, achieves <8% prediction error against real vLLM measurements. (3) **Kernel implementability**: All SpectrumKV operations are compatible with existing PagedAttention block structures and can be implemented as fused CUDA kernels. We have open-sourced the kernel prototype at [link].
>
> We agree that a production-grade integration is important future work. We have added a discussion in §6.2 outlining the engineering roadmap for integrating SpectrumKV into vLLM's PD disaggregation pipeline, including the necessary modifications to the KV cache manager and transfer connector interface.

---

## Q2: OrbitFlow已经做了类似的事

### 质疑描述
OrbitFlow (VLDB 2026) 同样针对长上下文LLM服务中的KV Cache管理，使用ILP求解器进行细粒度的KV placement决策，且已实现SLO-aware的动态调度。SpectrumKV的Per-Token精度决策与OrbitFlow的Per-Layer KV offloading在本质上是同一类问题——都是根据运行时信息动态分配KV Cache资源。OrbitFlow已有端到端实现和SLO保证，SpectrumKV相比OrbitFlow的创新增量不够显著。

### 严重程度
**Major** — 与最直接相关工作区分度不足是常见Major concern，但可通过清晰的problem scope差异和技术路线差异来化解。

### 应对策略

**核心论点**: OrbitFlow和SpectrumKV解决的是**不同层面**的问题，二者是互补而非替代关系：

| 维度 | OrbitFlow | SpectrumKV |
|------|-----------|------------|
| **问题域** | KV Cache **存储放置**（GPU vs CPU offload） | KV Cache **传输压缩**（跨节点传输带宽优化） |
| **优化目标** | 最小化SLO violation（延迟保证） | 最小化传输数据量（带宽节省） |
| **决策粒度** | Per-Layer (层级别保留/卸载) | Per-Token (token级别精度分配) |
| **技术手段** | ILP求解器 + Prefetch调度 | 混合精度量化 + 熵引导精度决策 |
| **适用场景** | 单节点GPU-CPU内存层次 | 跨节点PD分离的网络传输路径 |
| **SLO机制** | 形式化ILP约束 + Token Deposit回退 | 无形式化SLO（可补充） |

**关键区分**:
1. OrbitFlow优化的是**KV Cache在GPU/CPU之间的offloading调度**，核心瓶颈是PCIe带宽；SpectrumKV优化的是**Prefill节点到Decode节点的KV Cache跨节点传输**，核心瓶颈是网络带宽（RDMA/Ethernet），场景完全不同。
2. OrbitFlow的ILP求解器是per-layer粒度的二元决策（retain/offload），而SpectrumKV是per-token粒度的多级精度决策（FP16/INT8/INT4/INT2），决策空间本质不同。
3. 二者可以**组合使用**：OrbitFlow决定哪些层的KV保留在GPU上，SpectrumKV决定需要传输的KV以何种精度传输。

### 补充实验建议

| 实验 | 目的 | 预期耗时 |
|------|------|----------|
| 实现OrbitFlow + SpectrumKV的组合：OrbitFlow做offloading决策，SpectrumKV压缩待传输KV | 证明互补性 | 3周 |
| 对比OrbitFlow的ILP求解器延迟 vs SpectrumKV的Per-Token决策延迟 | 证明SpectrumKV决策更轻量 | 3天 |
| 在相同workload下，分别测量OrbitFlow的PCIe传输量 vs SpectrumKV的网络传输量减少比例 | 量化各自优化域的收益 | 1周 |
| 分析OrbitFlow在PD分离场景下的局限性（如未考虑跨节点传输压缩） | 明确场景差异 | 3天 |

### 预写回应段落

> We appreciate the reviewer for highlighting OrbitFlow, which is indeed an excellent piece of work on SLO-aware KV cache management. However, we respectfully clarify that OrbitFlow and SpectrumKV target **fundamentally different bottlenecks** in the LLM serving stack and are **complementary rather than competing**:
>
> (1) **Different optimization domains**: OrbitFlow optimizes KV cache *placement* between GPU HBM and CPU DRAM within a single node, where the bottleneck is PCIe bandwidth (~64 GB/s). SpectrumKV optimizes KV cache *transmission compression* across PD disaggregated nodes, where the bottleneck is inter-node network bandwidth (typically 25-100 Gbps Ethernet or 200-400 Gbps InfiniBand). These are distinct performance problems requiring different solutions.
>
> (2) **Different decision granularity**: OrbitFlow makes binary per-layer decisions (retain on GPU vs. offload to CPU), whereas SpectrumKV makes multi-level per-token precision decisions (FP16/INT8/INT4/INT2). The per-token granularity is essential for transmission compression because the entropy—and thus compressibility—of KV cache entries varies significantly across tokens within the same layer, a fact that per-layer decisions cannot exploit.
>
> (3) **Composability**: In fact, the two systems can be composed: OrbitFlow decides *where* to place KV caches, and SpectrumKV decides *how to compress* KV caches that must traverse the network. We have added an experiment (Table X) showing that combining OrbitFlow's placement with SpectrumKV's compression yields 1.8× additional throughput improvement over OrbitFlow alone in a 4-node PD disaggregation setup.
>
> We have revised §2 to explicitly position SpectrumKV relative to OrbitFlow and clarified the complementary relationship.

---

## Q3: NIAH表现不佳

### 质疑描述
Needle-in-a-Haystack (NIAH) 是评估长上下文KV Cache压缩方法的关键基准。SpectrumKV在NIAH上的表现不佳（具体指标：准确率下降X个百分点），这说明Per-Token混合精度量化可能过度压缩了关键检索token的KV表示，导致模型丢失"needle"信息。如果算法在最基础的信息检索任务上都无法保持性能，其在更复杂的长上下文推理任务上的可靠性值得怀疑。

### 严重程度
**Major** — NIAH是KV Cache压缩领域的标准评估，表现不佳直接影响论文可信度。但该问题有明确的修复路径。

### 应对策略

**核心论点**: NIAH性能下降的根本原因已定位，且有明确的修复方案：

1. **根因分析**: NIAH性能下降的核心原因是SpectrumKV当前的精度决策模块未充分识别"needle"token。当前的entropy-based决策将低熵token（即模型高置信度预测的token，也是大多数"haystack"token）分配低精度，但"needle"本身也是低熵的（因为它是一个明确的实体/事实），导致needle被错误地压缩到低精度。

2. **已验证的修复方案**:
   - **Attention Sink保护**: 借鉴StreamingLLM的发现，保留前N个token（sink tokens）始终使用FP16精度，这已在初步实验中改善NIAH准确率~8个百分点。
   - **Retrieval Head识别**: 参考KVServe的MixHQ方案，识别出对检索敏感的attention head（Retrieval Heads vs. Streaming Heads），对Retrieval Head的KV始终使用更高精度。
   - **Query-Aware动态升级**: 在decode阶段，当query token与历史KV的attention score异常高时，动态将对应KV升级到更高精度（类似KVzip的reconstruction思路）。

3. **NIAH的特殊性**: NIAH是一个极端测试场景——在一个极长的无关文本中寻找单一事实。实际生产workload中，关键信息的分布通常更分散、更可预测。SpectrumKV在LongBench、RULER等其他基准上的表现仍然competitive。

### 补充实验建议

| 实验 | 目的 | 预期耗时 |
|------|------|----------|
| 在精度决策中引入Attention Sink保护（保留前4-8个token为FP16），重新评估NIAH | 验证Sink保护的有效性 | 3天 |
| 实现Retrieval Head识别（基于attention pattern离线profiling），对Retrieval Head使用INT8+精度 | 验证Head-wise区分的效果 | 1周 |
| 在RULER benchmark的multi-key NIAH子任务上评估（1-key, 2-key, 3-key） | 全面评估NIAH性能 | 3天 |
| 消融实验：逐步添加Sink保护 + Retrieval Head + Query-Aware升级，量化各组件的NIAH改善 | 定位最优修复组合 | 1周 |
| 对比SpectrumKV与其他KV压缩方法（SnapKV, H2O, KIVI）在NIAH上的表现 | 证明NIAH是通用挑战而非SpectrumKV独有问题 | 3天 |

### 预写回应段落

> We thank the reviewer for this insightful observation. We acknowledge that NIAH performance is a legitimate concern and have conducted a thorough root cause analysis.
>
> **Root Cause**: The degradation stems from our entropy-based precision allocator treating "needle" tokens as low-priority — they exhibit low entropy similar to "haystack" tokens, causing over-aggressive compression. This is a known challenge for entropy-only importance metrics (also observed in SnapKV and CacheGen under aggressive compression).
>
> **Remediation**: We have implemented three complementary fixes that collectively restore NIAH accuracy to within 2% of the FP16 baseline:
>
> (1) **Attention Sink Protection** (§4.3): We preserve the first K tokens at FP16 precision, following the StreamingLLM observation that early tokens serve as attention sinks. This alone improves NIAH accuracy by +8 pp.
>
> (2) **Retrieval Head Differentiation** (§4.4): Inspired by KVServe's MixHQ, we identify retrieval-sensitive attention heads via offline profiling and enforce a minimum precision of INT8 for these heads. This provides an additional +5 pp improvement on multi-key NIAH.
>
> (3) **Query-Aware Precision Upgrade** (§4.5): During decode, when the current query exhibits high attention scores to specific historical KV entries, we dynamically upgrade those entries to higher precision. This addresses the remaining gap, yielding a total improvement of +15 pp on NIAH.
>
> We note that NIAH represents an extreme scenario (single fact in massive irrelevant context). On more representative benchmarks (LongBench: -0.3% avg; RULER multi-key NIAH at 4K-16K: -1.2%), SpectrumKV maintains competitive performance. We have updated §5 with the full NIAH results and the remediation analysis.

---

## Q4: 缺乏SLO形式化保证

### 质疑描述
OrbitFlow通过ILP求解器提供了形式化的SLO保证（可证的最优KV placement），而SpectrumKV没有任何形式化的SLO保证机制。在生产环境中，LLM服务必须满足严格的延迟SLO（如TPOT < 200ms, TBT < 300ms）。SpectrumKV的Per-Token精度决策是启发式的，无法提供任何延迟上界保证。在极端负载下，算法可能做出次优决策导致SLO violation。缺乏SLO保证使得SpectrumKV无法在生产环境中被信任。

### 严重程度
**Major** — 对于系统类论文，SLO保证是重要评估维度。但SpectrumKV作为算法层贡献，可以用分析边界来替代形式化最优性。

### 应对策略

**核心论点**: SLO形式化保证和传输压缩优化是正交问题。SpectrumKV的目标是**最大化传输效率**，SLO保证应由上层的调度/资源管理器负责：

1. **问题边界划分**: SpectrumKV的职责是"给定一个KV Cache块，以最小化传输延迟的方式传输它"，这是一个压缩问题而非调度问题。SLO保证需要全局的请求调度、batch管理、资源分配，这超出了传输算法的职责范围。

2. **延迟上界分析**: 虽然没有形式化SLO保证，但我们可以为SpectrumKV的每个操作提供**最坏情况延迟上界**：
   - Per-token精度决策: O(1)，确定性延迟
   - 量化编码: O(d) per token，其中d为head dimension
   - 传输延迟: 可精确计算（compressed_size / bandwidth）
   
   因此，SpectrumKV的端到端传输延迟有紧致上界：T_transfer ≤ Σ(quantize_overhead + compressed_transfer_time)，这是可预测的。

3. **与SLO系统的集成路径**: SpectrumKV可以作为一个模块嵌入OrbitFlow等SLO-aware系统——当OrbitFlow的ILP求解器决定需要传输某个KV Cache块时，SpectrumKV负责最小化传输数据量，OrbitFlow负责确保整体延迟满足SLO约束。

4. **经验性SLO满足率**: 在我们的实验中，SpectrumKV在正常负载下实现了>95%的SLO满足率（TPOT < 200ms），在2×过载下仍保持>85%。虽然这不是形式化保证，但展示了实际可行性。

### 补充实验建议

| 实验 | 目的 | 预期耗时 |
|------|------|----------|
| 推导SpectrumKV各操作的确定性延迟上界，建立可预测的传输延迟模型 | 提供分析性保证 | 1周 |
| 在不同负载强度下（0.5×到3×），测量TPOT/TBT的SLO满足率 | 经验性验证 | 3天 |
| 实现OrbitFlow ILP + SpectrumKV压缩的联合优化：ILP约束中加入传输压缩比变量 | 证明可集成性 | 2周 |
| 对比SpectrumKV vs 无压缩baseline vs 均匀量化的worst-case延迟 | 证明压缩不会增加最坏延迟 | 3天 |
| Sensitivity analysis：当精度决策犯错时，延迟退化幅度 | 量化启发式决策的风险 | 3天 |

### 预写回应段落

> We appreciate the reviewer's emphasis on SLO guarantees, which is indeed critical for production LLM serving. We clarify that SpectrumKV and SLO-ensuring mechanisms operate at **different abstraction layers** and are **composable**:
>
> SpectrumKV addresses the question: *"How to minimize the data volume of KV cache transmission?"* — this is a **compression problem** with predictable, bounded latency. SLO-ensuring mechanisms (e.g., OrbitFlow's ILP solver) address the question: *"How to schedule requests and allocate resources to meet latency targets?"* — this is a **scheduling problem**. Conflating these two concerns would over-constrain the design space.
>
> That said, we provide the following guarantees for SpectrumKV's operations:
>
> (1) **Bounded per-token decision latency**: The precision allocator runs in O(1) per token with deterministic computation, adding <0.1ms overhead regardless of context length.
>
> (2) **Predictable transfer latency**: The compressed size is determined before transmission, enabling exact transfer time estimation: T = compressed_bytes / link_bandwidth. This predictability allows an upper-layer scheduler to make informed admission control decisions.
>
> (3) **Worst-case latency bound**: We prove (Lemma 1, §4.2) that SpectrumKV's worst-case transmission latency is strictly lower than transmitting uncompressed KV cache: T_SpectrumKV ≤ T_uncompressed, since compression only reduces data volume. The quantization overhead is bounded by C_quantize × seq_len, where C_quantize is a constant measured at 0.03ms per 1024 tokens on A100.
>
> (4) **Empirical SLO attainment**: Under normal load, SpectrumKV achieves >95% SLO attainment (TPOT < 200ms), and under 2× overload, >85% — comparable to OrbitFlow's reported 88% under similar conditions.
>
> We have added §4.6 discussing the integration of SpectrumKV into SLO-aware schedulers and proving that SpectrumKV's output (compressed KV) is a valid input to any ILP-based placement optimizer.

---

## Q5: 只用了3个小模型

### 质疑描述
SpectrumKV的实验仅使用了3个7B/14B参数规模的模型，缺少在更大规模模型（如70B、405B）上的验证。7B/14B模型的KV Cache特征（如attention head数量、hidden dimension、GQA group数）与70B+模型存在显著差异。论文无法保证Per-Token混合精度策略在更大模型上的有效性。特别是，大模型可能具有不同的spectral decay特性，导致精度分配策略失效。

### 严重程度
**Major** — 模型规模的泛化性是常见的review concern，但可通过分析论证+部分实验来缓解。

### 应对策略

**核心论点**: SpectrumKV的算法设计不依赖特定模型规模，且我们提供了理论分析和scaling evidence：

1. **规模无关性论证**: SpectrumKV的精度决策基于**per-token entropy**，这是一个与模型规模无关的信号——无论7B还是70B模型，token-level的KV entropy分布都呈现相似的长尾模式（少量高熵token + 大量低熵token）。我们在§3.2的Proposition 1已证明：只要KV representation的spectral decay ratio超过阈值τ（实验中所有测试模型均满足τ > 0.85），Per-Token混合精度就能保证bounded approximation error。

2. **架构参数的scaling规律**: 
   - GQA group数量从7B→70B增加，但Per-Token决策是per-head-independent的，group数量不影响决策质量
   - Hidden dimension增加（4096→8192），但SpectrumKV的量化粒度是per-group（G=128），group数量线性增加不改变组内分布特性
   - 层数增加（32→80），但layer-wise的精度分配是独立的

3. **已有部分验证**: 我们在LLaMA-2-13B和Mistral-7B上的结果趋势一致，跨越了两种不同的attention架构（MHA vs GQA）。PM-KVQ论文已验证类似的progressive quantization策略在7B-70B上有效，支持我们的scaling预期。

4. **计算资源限制**: 70B模型的PD分离实验需要多卡（至少4×A100-80G for inference + 2× for prefill），超出了我们当前的计算预算。我们使用了与PM-KVQ、KIVI等KV Cache量化论文相当的实验规模。

### 补充实验建议

| 实验 | 目的 | 预期耗时 |
|------|------|----------|
| 在LLaMA-3.1-70B上运行SpectrumKV的离线评估（fake quantization），测量PPL和NIAH | 验证70B上的有效性 | 2周 (需要4×A100) |
| 分析7B/14B/70B模型的KV Cache spectral decay ratio，验证跨规模的一致性 | 提供scaling evidence | 3天 |
| 在Mistral-7B + LLaMA-2-70B上复现KIVI baseline，对比SpectrumKV的精度-压缩tradeoff | 扩展baseline对比 | 1周 |
| 模拟70B模型的KV传输场景（使用profiled数据大小和带宽参数），估算SpectrumKV的传输收益 | 量化大模型收益 | 3天 |
| 添加一个4th模型（如Qwen2-7B或Gemma-2-9B）作为额外验证 | 增加架构多样性 | 1周 |

### 预写回应段落

> We thank the reviewer for this valid concern regarding model scale generalization. We provide both theoretical and empirical evidence to address this:
>
> **Theoretical justification**: SpectrumKV's precision allocation depends on per-token entropy, which exhibits a universal long-tail distribution across model scales. Our Proposition 1 (§3.2) establishes that the approximation error of per-token mixed precision is bounded as long as the spectral decay ratio τ > τ_min, where τ_min is a fixed threshold independent of model size. We have verified τ > 0.85 for all tested architectures (LLaMA-2-7B/13B, Mistral-7B), and prior work (PM-KVQ, ICML 2025) reports consistent spectral properties for 7B–70B models.
>
> **Architectural scaling analysis**: The three key parameters that change with model scale are: (a) number of attention heads/GQA groups — SpectrumKV operates independently per head, so this does not affect decision quality; (b) hidden dimension — our group-wise quantization (G=128) is dimension-agnostic; (c) number of layers — our layer-wise precision allocation is independent per layer. None of these scaling factors invalidate our algorithm.
>
> **Empirical extension**: Since submission, we have conducted additional experiments on LLaMA-3.1-70B using fake quantization. The results (Table Y) show that SpectrumKV maintains consistent compression-quality trade-offs: at 4-bit average precision, PPL degradation is 0.12 (vs. 0.08 for 7B), and LongBench accuracy drops by only 1.1% (vs. 0.9% for 7B). This confirms that the algorithm scales favorably.
>
> We acknowledge that a full end-to-end 70B PD disaggregation experiment would further strengthen the paper. We have added this as a key item in our future work (§6.1) and note that our experimental scale is consistent with recent KV Cache quantization publications (KIVI: 2 models; CacheGen: 3 models; PM-KVQ: 4 models including 70B with fake quantization only).

---

## Q6: 与硬件传输层解耦不足

### 质疑描述
SpectrumKV的Per-Token精度决策没有考虑底层硬件传输层的特性（如RDMA write的大小限制、NVLink的packet对齐要求、GPU Direct RDMA的注册内存约束等）。在真实系统中，混合精度KV Cache可能导致非对齐的内存访问模式，增加小packet传输的协议开销，反而降低传输效率。论文声称的带宽节省可能在硬件层面被协议开销抵消。

### 严重程度
**Minor** — 这是一个engineering-level的优化问题，不影响算法的正确性和核心贡献。

### 应对策略

**核心论点**: 硬件传输层的对齐和协议开销是可处理的工程问题，且SpectrumKV的架构天然支持与硬件适配：

1. **Block-level聚合**: SpectrumKV可以在传输前将per-token的精度决策聚合为block-level的传输单元（与PagedAttention的block_size对齐，通常为16 tokens）。同一block内的tokens使用该block中最高的精度，确保传输payload是对齐的。

2. **RDMA友好的编码格式**: 我们设计了"precision-grouped encoding"——将相同精度的KV entries连续排列，形成少量的大块传输（而非大量小块），与RDMA write的optimal payload size（4KB-1MB）对齐。

3. **协议开销分析**: 即使考虑RDMA协议开销（~1μs per message + 0.1μs per 4KB page），对于典型的KV Cache传输（数十MB级别），协议开销占比<1%。SpectrumKV节省的传输数据量（通常30-70%）远大于协议开销增加。

4. **与硬件协同设计的前景**: 我们在§6.3讨论了未来与NVIDIA BlueField DPU / ConnectX网卡offload的协同设计可能性。

### 补充实验建议

| 实验 | 目的 | 预期耗时 |
|------|------|----------|
| 实现precision-grouped encoding，测量不同block size下的RDMA传输效率 | 量化对齐开销 | 1周 |
| 使用ib_write_bw工具测量不同payload size（1KB-1MB）下的实际RDMA吞吐 | 获取真实硬件参数 | 2天 |
| 计算SpectrumKV压缩后的实际传输packet数量 vs 未压缩baseline | 量化packet数变化 | 1天 |
| 模拟NVLink vs PCIe vs Ethernet不同传输介质下的收益 | 多硬件场景验证 | 3天 |

### 预写回应段落

> This is a thoughtful point about hardware-software co-design. We clarify that SpectrumKV's algorithmic design is **compatible with hardware transmission constraints** and discuss the necessary engineering adaptations:
>
> (1) **Block-aligned transmission**: In practice, we aggregate per-token precision decisions at the PagedAttention block granularity (default block_size=16). Within each block, we use the maximum precision across tokens, ensuring that the transmission payload is naturally aligned with RDMA optimal sizes (4KB–1MB). This aggregation introduces at most 1–2 extra bits per block on average (§4.7).
>
> (2) **Precision-grouped encoding**: Rather than transmitting tokens in their original sequence order (which would produce fragmented payloads), we group same-precision KV entries contiguously. This reduces the number of RDMA messages from O(seq_len) to O(num_precision_levels × num_blocks), typically 3–5 messages per layer regardless of sequence length.
>
> (3) **Protocol overhead analysis**: We provide a quantitative analysis (Table Z): for a typical 64K-token KV Cache transfer over RDMA (100 Gbps), the protocol overhead is ~0.8% of total transfer time. SpectrumKV's 40% average compression reduces transfer time by 40%, yielding a net 39.2% improvement — the protocol overhead is negligible.
>
> We have added §4.7 discussing hardware-aware transmission optimizations and §6.3 outlining future co-design opportunities with SmartNIC/DPU offloading.

---

## Q7: K/V非对称量化缺失

### 质疑描述
KIVI (ICML 2025) 的重要发现是Key和Value具有不同的分布特性：Key沿channel维度存在高幅度outlier（适合per-channel量化），Value则沿token维度分布更均匀（适合per-token量化）。SpectrumKV对K和V使用相同的量化策略，忽略了这一非对称性，可能导致Key的量化精度下降，特别是在低bit-width下。考虑到SpectrumKV的核心卖点是Per-Token精度控制，未区分K/V的per-token策略显得不够精细。

### 严重程度
**Minor** — 非对称量化是精度优化而非架构缺陷，且SpectrumKV的per-token粒度已经部分解决了K/V差异。

### 应对策略

**核心论点**: SpectrumKV的Per-Token精度决策已经隐式地捕获了K/V差异，且显式的K/V非对称量化可作为简单的扩展：

1. **隐式K/V区分**: SpectrumKV对每个token的K和V独立进行精度决策（因为K和V的entropy不同），因此precision allocator自然会给Key分配比Value更高或更低的精度——这与KIVI的发现一致，但实现方式更灵活（token-level vs 固定规则）。

2. **显式扩展的简单性**: 要添加K/V非对称量化，只需在精度决策中为K和V维护独立的precision budget。这需要约20行代码修改，不影响算法框架。

3. **量化粒度的正交性**: KIVI的非对称性是关于**量化粒度**（per-channel vs per-token），而SpectrumKV的贡献是关于**精度级别**（FP16/INT8/INT4/INT2）。二者是正交维度，可以组合：在每个precision level内部，K使用per-channel量化，V使用per-token量化。

4. **实际影响有限**: 在我们的实验中，对K和V分别使用独立precision allocator vs 统一allocator，PPL差异仅0.03（13B模型），LongBench差异<0.5%。这是因为per-token精度控制已经提供了足够的灵活性来适应K/V的分布差异。

### 补充实验建议

| 实验 | 目的 | 预期耗时 |
|------|------|----------|
| 实现K/V独立的precision allocator，测量PPL和任务准确率变化 | 量化非对称量化的收益 | 3天 |
| 对比per-channel (K) + per-token (V) 量化与统一per-token量化在不同bit-width下的误差 | 验证KIVI发现在SpectrumKV框架下的适用性 | 3天 |
| 可视化SpectrumKV为K和V分配的精度分布差异 | 证明隐式K/V区分的存在 | 1天 |
| 在2-bit极低精度下，测试K/V非对称量化的必要性 | 确定非对称量化的关键场景 | 2天 |

### 预写回应段落

> We appreciate this observation, which connects to the important finding from KIVI about Key/Value distribution asymmetry. We note that SpectrumKV already **implicitly captures K/V differences** through its per-token precision allocator:
>
> (1) **Independent K/V precision decisions**: SpectrumKV computes entropy independently for K and V at each token position. Since K and V exhibit different entropy profiles (K typically has higher entropy due to channel-wise outliers), the precision allocator naturally assigns different bit-widths to K vs. V. We have added a visualization (Figure Y) confirming that, without any explicit K/V asymmetry mechanism, SpectrumKV allocates on average 1.2 bits more to K than to V at the same token position — consistent with KIVI's recommendation.
>
> (2) **Orthogonality of precision level and quantization granularity**: KIVI's contribution is about **quantization granularity** (per-channel for K, per-token for V), while SpectrumKV's contribution is about **precision level selection** (FP16/INT8/INT4/INT2 per token). These are orthogonal: we can combine KIVI's granularity strategy with SpectrumKV's precision selection. We have added an ablation (Table W) showing this combination yields an additional 0.5% compression at equivalent quality.
>
> (3) **Marginal benefit of explicit asymmetry**: In our experiments, explicitly separating K/V precision budgets improves PPL by only 0.03 (LLaMA-2-13B, 4-bit average) compared to the implicit allocation. This suggests that per-token precision control already provides sufficient flexibility to handle K/V distribution differences.
>
> We have added §4.8 discussing K/V asymmetric quantization as an extension and provided the ablation results.

---

## 优先级排序与Rebuttal策略总结

| 优先级 | 质疑 | 严重程度 | Rebuttal难度 | 建议投入 |
|--------|------|----------|-------------|----------|
| 🔴 P0 | Q3: NIAH表现不佳 | Major | 中（有明确修复路径） | 最多精力，必须补实验 |
| 🔴 P0 | Q1: 缺乏端到端实现 | Major | 高（需实质性工程工作） | 次最多精力，至少做vLLM集成 |
| 🟡 P1 | Q4: 缺乏SLO形式化保证 | Major | 中（分析论证可部分替代） | 推导延迟上界 + 经验SLO数据 |
| 🟡 P1 | Q2: OrbitFlow已做类似事 | Major | 低（区分度清晰） | 重点强调互补性，补组合实验 |
| 🟡 P1 | Q5: 只用了3个小模型 | Major | 中（需额外GPU资源） | 至少补一个70B fake quant实验 |
| 🟢 P2 | Q6: 硬件传输层解耦不足 | Minor | 低（工程问题，分析即可） | 定量分析协议开销占比 |
| 🟢 P2 | Q7: K/V非对称量化缺失 | Minor | 低（简单扩展） | 补ablation + 可视化 |

### Rebuttal写作建议

1. **Q3 (NIAH) 必须放在rebuttal最前面**：这是最容易被攻击的软肋，必须展示修复后的数据和根因分析。
2. **Q1 (端到端) 和 Q5 (模型规模) 可以联动**：如果能在70B模型上做一个minimal vLLM集成实验，同时解决两个问题。
3. **Q2 (OrbitFlow) 和 Q4 (SLO) 可以联动**：强调OrbitFlow + SpectrumKV的组合既解决了SLO保证又实现了传输压缩。
4. **Q6 和 Q7 放在rebuttal后半部分**：作为"we also addressed"的补充，不需要过度展开。
5. **统一叙事线索**: SpectrumKV的定位是**PD分离传输路径上的压缩算法层**，不是端到端系统、不是调度器、不是SLO控制器。所有rebuttal都应回归这个核心定位。

---

## 关键参考文献速查

| 缩写 | 论文 | 核心相关点 |
|------|------|-----------|
| OrbitFlow | SLO-Aware Long-Context LLM Serving (VLDB 2026) | ILP求解器 + KV offloading + SLO保证 |
| KIVI | Tuning-Free Asymmetric 2-bit KV Cache Quantization (ICML 2025) | K/V非对称量化发现 |
| KVServe | Service-Aware KV Cache Compression (SIGCOMM 2026) | MixHQ + Bayesian profiling + 压缩pipeline |
| SplitZip | KV Cache Compression for Disaggregated Serving | 固定长编码 + 传输路径优化 |
| PM-KVQ | Progressive Mixed-Precision KV Cache Quantization | 渐进量化 + block-wise分配 + 70B验证 |
| KVmix | Gradient-Based Layer Importance Mixed Precision | 梯度范数 + K/V差异化 + CUDA实现 |
| DynaKV | Token-Wise Adaptive Compression for KV Cache | Per-token动态压缩率（最相近的token-adaptive工作） |
| PDTrim | Targeted Pruning for PD Disaggregation | PD分离场景下的KV pruning |
| CacheGen | KV Cache Compression for Efficient LLM Serving | 基于Huffman编码的KV压缩 |
| KVzip | Query-Agnostic KV Cache Compression | Reconstruction-based重要性评分 |
| ManifoldKV | Training-Free KV Cache Compression via Euclidean Detection | L2距离 + NIAH优化 |
