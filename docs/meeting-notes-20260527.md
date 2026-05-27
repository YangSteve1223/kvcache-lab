# KV Cache 新项目头脑风暴会议纪要

> **日期**: 2026-05-27  
> **参与者**: 系统架构师 / 研究前沿分析师 / 工程可行性评估师 / 创新点挖掘师  
> **决议**: 单开新项目，主攻PD分离场景下的KV Cache优化

---

## 一、四方共识

| 维度 | 共识 |
|------|------|
| **主攻方向** | PD分离 + KV Cache压缩联合优化（非纯压缩、非纯调度） |
| **项目独立性** | 独立仓库，不覆盖LLM-Serving-Lab |
| **技术栈** | 继续TypeScript，通过REST API对接vLLM/SGLang |
| **验证方式** | 仿真为主（自研模拟器），DeepSeek API辅助，无需GPU |
| **目标周期** | 3-4个月出论文/简历成果 |

---

## 二、研究前沿（分析师报告）

### 2025-2026最热子方向
1. **PD分离+自适应KV压缩**（最热）— KVServe (SIGCOMM 2026) 9.13倍JCT加速；SplitZip 618 GB/s压缩吞吐
2. **端到端量化压缩** — TurboQuant (ICLR 2026) 6倍压缩零精度损失；LeanKV 3-11倍压缩
3. **SLO感知长上下文管理** — OrbitFlow (VLDB 2026) P95延迟降低38%

### 必读论文
| 论文 | 会议/来源 | 关键点 |
|------|----------|--------|
| KVServe | SIGCOMM 2026 | PD+压缩里程碑，服务感知压缩 |
| SplitZip | 2026 | PD无损压缩，618 GB/s吞吐 |
| KVzip | NeurIPS 2025 Oral | 查询无关压缩理论基础 |
| OrbitFlow | VLDB 2026 | SLO感知框架 |
| TurboQuant | ICLR 2026 | 端到端量化，6倍零损压缩 |
| LeanKV | 2026 | 异质量化+每头动态稀疏 |
| Marconi | 2026 | RadixTree+FLOP-aware驱逐最新实践 |

### 核心创新机会
PD分离场景下，KV压缩与传输的**联合优化**是明显空白：
- KVServe做了服务感知压缩，但通信带宽/延迟建模仍有空间
- SplitZip做无损压缩，但**传输时机+压缩粒度**的联合优化未探索
- 可结合排队论做理论分析（契合主人通信工程背景）

---

## 三、创新点（挖掘师报告）

### 🔥 创新点1：PD-Aware KV Compression（⭐⭐⭐⭐⭐ 最推荐）
**论文标题**: 《PDSplit: Prefill-Decoder Disaggregation Aware KV Cache Compression》

- P端可采用更激进压缩（KV会传输到D端，压缩损失被传输效率补偿）
- D端保持更多KV支持迭代生成，考虑KV聚合批处理效率
- **首个PD分离场景下的压缩策略设计空间探索**
- 验证：用现有PD模拟器注入不同压缩比，测量TTFT+ITL的Pareto前沿

### 🔥 创新点2：Task-Aware Layer Budget Allocation（⭐⭐⭐⭐）
**论文标题**: 《TaskSpecKV: Task-Adaptive Layerwise KV Cache Budget Allocation》

- 不同任务类型（教育/代码/对话）attention分布模式不同
- 离线构建任务-层注意力模式知识库，推理时自适应选择预算策略
- 可与创新点1组合，形成"PD分离+任务自适应"完整叙事

### 创新点3：Speculative Draft KV Management（⭐⭐⭐）
- Draft KV共享、自适应gamma、Pre-commit压缩
- TransKV已占坑，可作为后续扩展

### 创新点4：InfoPD信息论框架（⭐⭐⭐）
- 扩展CAPKV目标函数：max I(retained KV → gen) - λ × transfer_cost
- 理论价值高但风险大，适合组合使用而非单独出论文

### 推荐组合
**创新点1 + 创新点2** = "PD分离场景下任务自适应的KV压缩"
- 工程落地价值（与vLLM/SGLang对标）
- 学术创新（任务感知机制）
- 3-4个月可行

---

## 四、系统架构（架构师报告）

### 项目命名
- **推荐**: `kvopt` 或 `kvcache-lab`
- GitHub: `YangSteve1223/kvopt` 或 `YangSteve1223/kvcache-lab`

### 目录结构（Monorepo）
```
kvopt/
├── packages/
│   ├── simulator/          # 仿真核心（fork自LLM-Serving-Lab，精简）
│   ├── compression/        # KV压缩算法（KVzip/PDTrim/TaskAware）
│   ├── routing/            # Cache-Aware Router（PD路由+压缩策略联合）
│   └── experiments/        # 实验框架（基准+对比+统计）
├── scripts/                # 实验脚本
├── docs/                   # 文档
└── reports/                # 实验报告（.gitignore）
```

### 核心架构
```
┌─────────────────────────────────────────────┐
│      Compression Orchestrator (插件式)       │
│   KVzip + PDTrim + TaskAware 可组合叠加     │
└─────────────────┬───────────────────────────┘
                  │
┌─────────────────▼───────────────────────────┐
│          Cache-Aware Router                 │
│   根据缓存命中 + 任务类型 → 路由+压缩策略    │
└─────────────────┬───────────────────────────┘
                  │
         ┌───────┴───────┐
         ▼               ▼
    ┌─────────┐    ┌─────────┐
    │ Prefill │    │ Decode  │
    │  Node   │    │  Node   │
    └────┬────┘    └────┬────┘
         │               │
         └───────┬───────┘
                 ▼
        ┌────────────────┐
        │ Unified Cache  │  ← Radix/Hierarchical 共享
        └────────────────┘
```

### 代码复用策略
| 来源 | 模块 | 方式 |
|------|------|------|
| LLM-Serving-Lab | cache/* (6个模块) | 直接fork，精简无关代码 |
| LLM-Serving-Lab | EnhancedPDServingSimulator | fork，扩展KV压缩参数 |
| LLM-Serving-Lab | CacheExperimentRunner | fork，扩展压缩对比维度 |
| 新写 | Compression Orchestrator | 插件式，支持多压缩算法 |
| 新写 | Cache-Aware Router | 联合路由+压缩决策 |
| 新写 | TaskAware Budget Allocator | 任务自适应层预算分配 |

---

## 五、工程可行性（评估师报告）

### 难度评估
| 方向 | 难度 | 风险 | 说明 |
|------|------|------|------|
| PD分离+KV迁移 | ⭐⭐⭐ | 中 | 现有模拟器可扩展，关键是建模准确性 |
| 层次化缓存+自适应压缩 | ⭐⭐⭐ | 中 | 框架已有，策略算法需设计 |
| 在线KV驱逐策略 | ⭐⭐ | 低 | LRU/LFU已有，可扩展 |
| KV压缩算法核心 | ⭐⭐⭐⭐⭐ | 高 | 需信号处理知识，但可简化实现 |

### 无GPU验证策略
1. **仿真验证（主路径）**: CacheExperimentRunner对比不同策略token hit rate/TTFT
2. **DeepSeek API**: prompt prefix共享率测试、端到端延迟基准（KV内部指标不可测）
3. **Vidur**: 无KV Cache专用模块，仅辅助验证batch scheduling

### 关键风险
| 风险 | 缓解 |
|------|------|
| Python能力不足 | 核心逻辑保持TS，必要时纯TS复现算法 |
| 仿真可信度 | 用真实trace(ShareGPT)验证，对比公开benchmark |
| 时间不足 | 优先PD+KV压缩，层次化可后续迭代 |
| 压缩效果不达预期 | 提前小规模验证，设置明确评估指标 |

---

## 六、执行计划

### Month 1: 基础稳固 + 方向验证
- [ ] 创建新仓库 `kvopt`，fork现有缓存模块
- [ ] 修复RadixTree typo等已知bug
- [ ] 实现Compression Orchestrator插件框架
- [ ] 基准实验：cache on/off、eviction策略对比
- [ ] 精读KVServe/SplitZip/KVzip论文

### Month 2: 核心实现
- [ ] 实现PD-Aware KV压缩策略（P端激进/D端保守）
- [ ] 扩展PD模拟器支持KV压缩参数
- [ ] 实现Task-Aware Budget Allocator
- [ ] 中期实验报告：Pareto前沿可视化

### Month 3: 优化迭代
- [ ] Cache-Aware Router联合优化
- [ ] 多场景对比实验（教育/代码/对话workload）
- [ ] DeepSeek API端到端验证
- [ ] 完整实验报告

### Month 4: 收敛输出
- [ ] 论文写作（Introduction + Related Work + Method + Evaluation）
- [ ] 代码整理与文档
- [ ] arXiv投稿 + 开源代码
- [ ] 简历更新

---

## 七、决策清单（待主人确认）

1. **项目命名**: `kvopt` vs `kvcache-lab` vs 其他？
2. **创新点组合**: 确认走"PD-Aware + Task-Aware"路线？
3. **仓库结构**: Monorepo（packages/）还是扁平结构？
4. **首个milestone**: 直接开搞还是先精读KVServe/SplitZip？
5. **与原项目关系**: 是否需要保持API兼容，还是完全独立？
