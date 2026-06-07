# 论文扩充审核标准

## 审核时机
子session 7648145137620615439 (expand-paper-v2) 完成后，立即启动审核agent。

## 审核对象
- `/app/data/所有对话/主对话/kvcache-lab/paper/main_v2.tex`
- `/app/data/所有对话/主对话/kvcache-lab/paper/spectrumkv_v2.bib`

## 审核维度（5项，每项1分，总分5分）

### 1. 数据真实性 (Data Integrity)
**通过标准**：所有数值可追溯到JSON源文件，无编造/修改/四舍五入失真
**检查方法**：
- 从tex中提取所有数值声明（PPL、cosine sim、NIAH success rate等）
- 与源JSON交叉验证（exp1_ppl_*.json, exp2_niah_*.json, exp3_quant_error_*.json, exp4_layer_budget_*.json）
- 特别检查：PPL曲线数据点、per-layer cosine值、NIAH depth成功率
**不通过处理**：逐条标注偏差，返工修正

### 2. 学术风格 (Academic Style)
**通过标准**：
- 无overclaiming：禁"首次""从根本上""重新定义""打破认知"
- 禁止词汇："v2""改进""沿用""优化版"指代SpectrumKV
- 无AI套话：禁三连排比、段尾机械升华、开头句式雷同
- 术语一致：SWS=Semantic Working Set, SpectrumKV=核心算法, budget=传输预算比例
- 不改变原文叙事风格
**不通过处理**：标注违规位置和替代建议，返工改写

### 3. 内容完整性 (Content Completeness)
**通过标准**：9项扩充任务全部存在：
1. PPL vs Budget曲线图（3模型×4方法，pgfplots折线图）
2. 消融实验小节（sink保护/重要性排序vs随机/2-tier vs 3-tier）
3. NIAH depth热力图（Qwen: Adaptive vs PDTrim, budget×depth矩阵）
4. 逐层INT4误差图（per-layer cosine similarity）
5. 理论分析扩充（softmax amplification lemma, 误差传播bound）
6. 与均匀量化分析对比
7. Discussion扩展（INT4根因/冷层fetch/互补性）
8. Related Work扩充（OrbitFlow/CacheBlend/SplitZip/LMCache/NVIDIA Dynamo/QuiP等）
9. System Architecture概念图
**不通过处理**：列出缺失项，返工补齐

### 4. LaTeX编译 (Compilation)
**通过标准**：pdflatex+bibtex编译无错误，生成PDF
**检查方法**：
```bash
cd /app/data/所有对话/主对话/kvcache-lab/paper/
pdflatex -interaction=nonstopmode main_v2.tex
bibtex main_v2
pdflatex -interaction=nonstopmode main_v2.tex
pdflatex -interaction=nonstopmode main_v2.tex
```
**不通过处理**：返工修编译错误

### 5. 叙事连贯性 (Narrative Coherence)
**通过标准**：
- 扩充内容与原文叙事风格一致
- 新旧内容之间过渡自然，不突兀
- 无重复论述
- 无自相矛盾（如数据与结论不一致）
**不通过处理**：标注矛盾/突兀处，返工调整

## 评分与处理
- 5/5：直接交付
- 4/5：Minor revision，主agent直接修后交付
- ≤3/5：Major revision，退回子session返工，修复后二次审核

## 关键数据源
- PPL数据：`/app/data/所有对话/主对话/spectrumkv_data/exp1_ppl_{mistral7b,gemma9b,qwen7b}.json`
- NIAH数据：`/app/data/所有对话/主对话/spectrumkv_data/exp2_niah_{mistral7b,gemma9b,qwen7b}.json`
- 量化误差：`/app/data/所有对话/主对话/spectrumkv_data/exp3_quant_error_{mistral7b,gemma9b,qwen7b}.json`
- 层预算：`/app/data/所有对话/主对话/spectrumkv_data/exp4_layer_budget_*.json`
- 全量数据汇总：`/app/data/所有对话/主对话/spectrumkv_data/ALL_EXPERIMENT_DATA.md`

## 原始论文（扩充基础）
- tex：`/app/data/所有对话/主对话/用户上传/main.tex`（798行）
- bib：`/app/data/所有对话/主对话/用户上传/spectrumkv.bib`（128行）


## 润色阶段（审核通过后执行）

### 流程
1. 审核通过(≥4分)后，以main_v2.tex为base，启动4个并行润色agent
2. 每个agent输出到独立路径：
   - Agent A → main_v2_polish_a.tex
   - Agent B → main_v2_polish_b.tex
   - Agent C → main_v2_polish_c.tex
   - Agent D → main_v2_polish_d.tex
3. 主agent对4个版本进行**段落级择优合并**：
   a. 将4个版本按section/paragraph对齐
   b. 逐段落对比4个版本的同一位置，从以下维度评分：
      - AI痕迹最少（三连排比/套话/段尾升华/句式雷同）
      - 学术表达最自然（不过度正式也不随意）
      - 叙事最流畅（过渡自然、论证紧凑）
      - 信息密度最高（没有waterfilling的废话）
   c. 每个段落选得分最高的版本
   d. **合并后校验**：
      - 语义完整性：每个段落都有且仅有一份，无遗漏
      - 语义去重：相邻段落不重复论述同一观点
      - 过渡一致性：段落间衔接自然，不因来源不同而割裂
      - 术语一致性：SpectrumKV/SWS/budget等跨段落统一
   e. 对过渡不自然处进行微调润色
4. 最终合并版本编译为main_v2_final.tex + PDF交付

### 4个润色Agent的差异化策略

**Agent A — 句式多样性专家**
- 核心目标：打破AI写作的句式模式
- 具体动作：
  - 扫描所有连续3句以上相同主语开头的段落，改写其中至少1句
  - 检查每段首句是否都是"We..."开头，替换1/3为其他结构
  - 将过长的复合句(>40词)拆分为2个短句
  - 将过短的碎片句合并
  - 确保每个section至少有1个反问句或否定前置句

**Agent B — 去套路化专家**
- 核心目标：消除AI写作的公式化表达
- 具体动作：
  - 删除所有"Moreover/Furthermore/Additionally/It is worth noting"等AI高频连接词，替换为更具体的逻辑连接
  - 删除段尾无实质信息的总结句（如"This demonstrates the effectiveness of..."）
  - 检查三连排比（A, B, and C结构连续出现），改写为不规则列举
  - 删除"It is important to note/It should be emphasized"等空壳强调
  - 确保每个claim都有具体数据支撑，不靠语气词增强

**Agent C — 精准学术写作专家**
- 核心目标：提高信息密度，消除waterfilling
- 具体动作：
  - 标记所有可删除但不影响语义的句子/从句，删除
  - 将描述性文字替换为定量陈述（"significant improvement" → 具体百分比）
  - 合并重复论述同一观点的段落
  - 确保Related Work每段末尾有明确的与本文对比，不是简单罗列
  - Discussion每段必须有新观点，不重复Results已说过的

**Agent D — 叙事节奏专家**
- 核心目标：让论文读起来像一个有观点的研究者在讲故事，而非模板填充
- 具体动作：
  - 检查section间过渡是否自然，每个section开头是否承接上文而非突兀切入
  - 确保关键发现用"意外/反直觉"框架呈现（如"Surprisingly, we find that..."）
  - 在适当位置加入"why"的解释，而非只报告"what"
  - 检查被动语态过度使用，将1/3转为主动
  - 确保figure/table引用融入论证流，不是孤立插入

### 润色通用规则（4个agent共享）
- 不改变任何数据、公式、图表内容
- 不改变论文结构和section顺序
- 不删除任何figure/table/equation
- 保持SpectrumKV/SWS/budget等术语一致性
- 禁止"v2/改进/沿用"指代SpectrumKV
- 禁止overclaiming
- 最终输出必须可pdflatex编译
