# Paper Draft: Runtime KV Memory Management for PD-Disaggregated LLM Serving

## 文件结构

```
kvcache-lab/paper/
├── main.tex        # 完整论文草稿 (LaTeX格式)
└── references.bib  # 参考文献列表
```

## 论文结构

### 已完成 Section

| Section | 状态 | 说明 |
|---------|------|------|
| Abstract | ✅ 完成 | 四步走：背景→问题→方案→量化结果 |
| Introduction | ✅ 完成 | 四段式：Motivation→Problem→Solution→Contributions |
| Background and Motivation | ✅ 完成 | 含2.1 PD分离、2.2 KV瓶颈、2.3 Locality观察 |
| Design | ✅ 完成 | SWS主贡献、TAA机制、Tiering设计 |
| Implementation | ✅ 完成 | vLLM集成、组件说明 |
| Evaluation | ⚠️ 部分完成 | 现有数据已填充，待补Long Context和Serving Benchmark |
| Related Work | ✅ 完成 | LLM Serving、KV优化、PD disaggregation |
| Conclusion | ✅ 完成 | 总结+未来方向 |

### TODO 标记（红色高亮）

以下位置有 `\textcolor{red}{TODO: ...}` 标记，需要后续补充：

1. **Figure占位符** (6个)
   - Figure 1: PD Disaggregation Architecture Overview
   - Figure 2: Attention Score Distribution Over Time
   - Figure 3: SWS System Architecture
   - Figure 4: SWS vs SWA Comparison
   - Figure 5: KV Re-fetch Pipeline
   - Figure 6-10: 实验图表

2. **实验数据** (Section 6.5, 6.6)
   - Section 6.5: Long Context Scaling - 需要8K/16K/32K/64K实验
   - Section 6.6: Serving Benchmark - 需要vLLM吞吐测试

3. **表格待填数据**
   - Table 4: ws=2048的PPL数据
   - Table 5: 32K context的PPL impact
   - Table 7-9: 多个TODO数据点

4. **其他**
   - Experimental Setup: 需要明确模型规格
   - Acknowledgments: 需要填写

## 核心卖点 (Key Selling Points)

### 1. 系统级发现
**Decode阶段存在极强的locality，大部分remote KV对decode不必要。**

- 类比OS working set概念
- 不同于NLP tricks，是systems insight

### 2. SWS ≠ Sliding Window Attention
- **SWS**: Runtime KV cache tiering（降级但仍可访问）
- **SWA**: Architecture change（token完全不可访问）
- 类比OS: hot/cold pages, tiered memory

### 3. TAA作为Enabling Mechanism
- Low-overhead locality-aware guidance
- Overhead < 0.04%
- Layer sensitivity: last 1/2 layers (14-27)

### 4. 实验验证
| 实验 | 结果 |
|------|------|
| G4 Memory Savings | 93.8%-99.8% 内存节省 |
| G6 Full OS Ablation | 50%内存预算下PPL仅-0.42% |
| G7 Concurrency | 8x 更多并发 (32K上下文) |

## 编译说明

```bash
cd kvcache-lab/paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

或者使用 latexmk：
```bash
latexmk -pdf main.tex
```

## 参考文献

共引用17篇论文，主要包括：
- vLLM (OSDI'24)
- DistServe (OSDI'24)
- Splitwise (ATC'24)
- Tetris (HotOS'24)
- Longformer (ICML'20)
- StreamingLLM (ICML'23)
- 等

## 写作风格

参考vLLM(SOSP), DistServe(OSDI), Splitwise(ISCA)：
- 主动语态（We propose, We observe）
- 一般现在时
- 量化优先
- OS类比贯穿全文
