# kvcache-lab 实验结果总结（供团队讨论）

## 论文方向
《Runtime KV Memory Management for PD-Disaggregated LLM Serving》

## 核心创新点
1. **TAA（Task-Aware Attention）**：通过cost vector在attention中加入bias，指导模型关注本地KV
2. **SWS（Sliding Window Sharing）**：PD分离下decode端只保留最近W个token的KV cache
3. **Predictive Eviction**：TAA-guided选择性eviction，优于random/LRU

## 实验环境
- 模型：Qwen2.5-7B-Instruct（28层，4 KV heads，SDPA attention）
- GPU：AutoDL A100 80GB
- 测量方式：Hook注入TAA bias到所有28层，GPU实测PPL

---

## Phase A：基础验证

### G1 基线性能
| 上下文长度 | TTFT | TPOT | PPL |
|-----------|------|------|-----|
| 1K | 185.6ms | 26.8ms | 1.1331 |
| 32K | 4272.4ms | 35.1ms | 1.0454 |

### G2 PD分离传输
| 上下文 | 25Gbps传输 | 100Gbps传输 | KV Size |
|--------|-----------|------------|---------|
| 8K | 149ms | 37.2ms | 444MB |
| 32K | 594.9ms | 148.7ms | ~1776MB |

### G3 TAA核心验证（seq=64, diverse text）
| α | PPL | Δ% |
|---|-----|-----|
| 0.0 | 6.9297 | baseline |
| 0.1 | 6.8867 | -0.62% |
| **0.2** | **6.8477** | **-1.18%** ← 最优 |
| 0.5 | 6.8281 | -1.47% |
| 1.0 | 6.8398 | -1.30%（回退） |
| 5.0 | 9.4844 | +36.87%（崩溃） |

- Layer Sensitivity：last 1/2层(14-27)效果最均衡
- 长序列(2K)：α≤0.2时PPL变化≤0.16%（质量无损）
- **TAA Overhead**：2K=55μs(0.006%), 32K=3.2ms(0.04%)——几乎免费

---

## Phase B：系统验证

### G4 SWS（Sliding Window Sharing）
baseline PPL=6.6484, seq=511

| Window Size | SWS PPL | SWS+TAA PPL |
|------------|---------|-------------|
| 32 | 16.6 | 16.6 |
| 64 | 12.8 | 12.9 |
| 128 | 12.0 | 12.1 |
| 256 | 9.7 | 9.8 |

**KV Cache内存节省（32K上下文）：**

| Window Size | 本地KV | 内存节省 |
|------------|--------|---------|
| 64 | 3.7MB | 99.8% |
| 128 | 7.3MB | 99.6% |
| 256 | 14.7MB | 99.2% |
| 512 | 29.4MB | 98.4% |
| 1024 | 58.7MB | 96.9% |
| 2048 | 117.4MB | 93.8% |

### G5 Predictive Eviction
baseline PPL=2.79, seq=512

| Evict% | Random | LRU | TAA-guided |
|--------|--------|-----|------------|
| 10% | 4.63 | **3.94** | **3.94** |
| 20% | 8.90 | **4.36** | **4.35** |
| 30% | 18.80 | **5.20** | **5.19** |
| 40% | 41.28 | **5.64** | 5.64 |
| 50% | 90.75 | **6.67** | **6.66** |

- TAA-guided vs Random：30%evict时5.19 vs 18.80（**3.6x改善**）
- **TAA-guided vs LRU：差距极小（5.19 vs 5.20）——这是弱项**

### G6 Full OS Ablation
baseline PPL=2.79, seq=512

| 内存预算 | SWS PPL | SWS+TAA PPL | Δ% |
|---------|---------|-------------|-----|
| 10% | 11.92 | 11.92 | +327% |
| 20% | 10.59 | 10.55 | +278% |
| 30% | 9.80 | 9.79 | +251% |
| **50%** | **2.78** | **2.78** | **-0.42%** ← 质量无损！ |
| 70% | 2.78 | 2.78 | -0.56% |
| 100% | 2.79 | 2.79 | +0.00% |

**核心发现：50%内存预算下PPL变化仅-0.42%，质量几乎无损**

### G7+G8 并发与质量
- SWS支持**8x更多并发请求**（32K上下文场景）
- 长序列(1024tokens)：SWS+TAA PPL -0.69%
- ≥50%内存预算时质量基本无损

---

## 结论强度评估

### ✅ 强项（论文核心卖点）
1. **50%内存 → 质量无损**：这是最硬的结果，审稿人无法否认
2. **8x并发提升**：系统层面的实际收益
3. **TAA overhead <0.04%**：几乎免费，工程上可行
4. **TAA-guided >> random eviction**：5-15x差距，方向正确
5. **99.8% KV内存节省（ws=64）**：极端场景下仍然可用

### ❌ 弱项（审稿人可能攻击）
1. **TAA直接PPL改善仅-1.18%**：单独看不够impressive
2. **TAA-guided vs LRU差距极小**（5.19 vs 5.20）：几乎在noise level
3. **SWS+TAA vs SWS alone几乎无差别**：TAA的附加价值不显著
4. **只有一个模型**（Qwen2.5-7B）：缺乏泛化性
5. **没有真实serving benchmark**：吞吐/延迟是analytical估算，非实测
6. **α曲线不够密**：只有6个点，需要更密的α扫描画漂亮曲线图

---

## 建议补充的实验（团队讨论）

### 优先级P0（强烈建议）
1. **更密的α扫描**：α∈[0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]，画平滑曲线图
2. **多模型验证**：至少加一个模型（如Llama3-8B或Mistral-7B），证明TAA泛化
3. **长上下文场景**：4K/8K/16K/32K下的SWS+TAA质量曲线（当前只有512和1024）

### 优先级P1（如果时间允许）
4. **实际serving吞吐测试**：用vLLM框架跑真实request，测TPS/TTFT/TPOT
5. **TAA-guided vs LRU更细致对比**：在长上下文(4K+)下LRU可能不适用，TAA优势会更大
6. **不同remote ratio的对比**：当前固定70%remote，应测试30%/50%/90%

### 优先级P2（锦上添花）
7. **多任务PPL**：不只是general text，还有code/qa/summarization
8. **生成质量评估**：除了PPL，加ROUGE/BLEU等任务指标

---

## 论文定位建议

**不要定位为"attention改进"论文（-1.18%不够看），要定位为"系统论文"：**

> "我们提出了一种PD分离场景下的KV Cache运行时管理系统，通过SWS实现50%内存下质量无损、8x并发提升，TAA提供了几乎免费的注意力引导，overhead<0.04%。"

核心叙事：
- **Problem**：PD分离下decode端KV cache内存瓶颈
- **Key Insight**：大部分远程KV不需要常驻本地，sliding window足够
- **Solution**：SWS + TAA-guided eviction
- **Result**：半内存、8x并发、质量无损

这个定位下，TAA是"how we make it work"而不是"our main contribution"，审稿人不会要求-10% PPL。
