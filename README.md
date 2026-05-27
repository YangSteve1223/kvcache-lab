# kvcache-lab

**Runtime KV Memory Management for PD-Disaggregated LLM Serving**

像操作系统管理内存一样管理KV Cache的生命周期——从创建、驻留、迁移、驱逐到召回。

## 核心思想

> KV cache should be managed as communication-aware runtime memory, rather than compressed as static context.

传统方法把KV Cache当作"静态上下文"来压缩，我们把它当作"运行时内存"来管理：
- **不是问"哪个token重要"**（importance estimation已被CapKV占据）
- **而是问"KV的系统行为是什么"**——何时被访问、在哪里存储、传输成本多少

## 系统架构

```
┌──────────────────────────────────────────────────┐
│           Global State Store (唯一数据源)         │
└──────────────────────────────────────────────────┘
  ↑ 写入        ↑ 写入        ↑ 写入        ↑ 写入
┌─────────┐ ┌─────────┐ ┌────────────┐ ┌─────────┐
│Semantic │ │  Reuse  │ │Communication│ │Placement│
│ Agent   │ │  Agent  │ │   Agent     │ │ Agent   │
└─────────┘ └─────────┘ └────────────┘ └─────────┘
                    ↓ 读取
             ┌─────────────────┐
             │ RuntimeScheduler│  max Q - λ₁L - λ₂M - λ₃T
             └─────────────────┘

底层压缩层：Phase-aware IB + R-D
```

### 核心创新

| 层级 | 组件 | 说明 |
|------|------|------|
| **Core Mechanism** | Transmission-Aware Attention | `score = relevance + α × runtime_bias`，attention偏好低成本KV |
| **Runtime Policy** | Semantic Working Set | 基于decode-time locality动态维护活跃KV子集 |
| **Runtime Policy** | Predictive Eviction | Belady-inspired reuse prediction，limited horizon(32 steps) |
| **Infrastructure** | Global State Store + Scheduler | Agent解耦通信，统一目标函数优化 |

### OS概念映射

| OS概念 | KV系统对应 |
|--------|-----------|
| Working Set | Semantic Working Set (活跃语义区域) |
| Cache Replacement | Predictive KV Eviction (reuse distance预测) |
| Virtual Memory | Hierarchical KV Memory (GPU/CPU/Remote分层) |
| Page Migration | KV Transfer & Placement (跨节点迁移) |
| Memory Access Cost | Transmission-Aware Attention (通信成本感知) |

## 项目结构

```
src/
├── core/          # 仿真引擎（PDSimulator, KVCacheManager, QualityModel）
├── compression/   # 压缩策略（None/Uniform/PD-Aware/Task-Aware/PD-Task-Aware）
│   └── algorithms/  # 真实压缩算法（KVQuantizer, AttentionPruner）
├── task/          # 任务感知（TaskClassifier, LayerBudgetAllocator, Profiles）
├── ib/            # Information Bottleneck（PhaseAwareIB, IBCompressor）
├── rd/            # Rate-Distortion（SemanticDistortion, AdaptiveTransmission, RDCompressor）
├── unified/       # IB+RD统一框架（SemanticRDFramework, IBRDCompressor）
├── agents/        # 4-Agent协同
│   ├── SemanticAgent.ts      # 语义区域识别 + Working Set计算
│   ├── ReuseAgent.ts         # reuse distance预测
│   ├── CommunicationAgent.ts # 传输成本评估 + TAA
│   └── PlacementAgent.ts     # KV分层放置 + 迁移队列
├── runtime/       # Runtime核心
│   ├── GlobalState.ts        # 全局状态存储（Agent间解耦）
│   └── RuntimeScheduler.ts   # 统一目标函数调度
├── baselines/     # 对标基线（KVServe, PDTrim简化版）
└── scheduling/    # SLO感知路由
experiments/       # 实验脚本（exp1-44）
logs/              # 实验日志
charts/            # 论文可视化
gpu-experiments/   # GPU验证实验（Python脚本）
data/              # 真实Workload数据
docs/              # 调研报告+新颖性验证
```

## 实验概览

### Phase 1: PD-Aware + Task-Aware（已完成）
- exp1-17: PD-Aware压缩、带宽敏感性、分类器增强、真实Workload、模型缩放、API验证、Baseline对标

### Phase 2: IB/R-D理论框架（已完成）
- exp18-25: Phase-aware IB（质量+12-21% over CapKV）、Phase-aware R-D（带宽节省97.2%）、统一框架

### Phase 3: Runtime KV Memory OS（已完成仿真，GPU验证进行中）
- exp32-39: 4-Agent实现、协调验证、Full OS端到端、Ablation Study
- exp40-44: 策略替换Ablation、多请求并发、长上下文缩放、带宽×策略矩阵、权重敏感性

### 关键结果

| 指标 | 数据 |
|------|------|
| Full OS vs Baseline 延迟 | 降低29% |
| TAA 高拥塞延迟节省 | 60% |
| Semantic Agent 质量提升 | +10% |
| Predictive Eviction 命中率 | 与Belady Oracle持平 |
| Phase-aware IB vs CapKV | 质量改进12-21% |
| Phase-aware R-D 带宽节省 | 97.2% |

## 快速开始

```bash
# 安装依赖
npm install

# 运行仿真实验
node --import tsx experiments/exp32-runtime-os-demo.ts

# 运行完整OS实验
node --import tsx experiments/exp39-full-runtime-os.ts

# GPU实验（需要2台GPU服务器）
cd gpu-experiments && bash setup.sh
```

## 技术栈

- **仿真**: TypeScript + Node.js 22，零GPU依赖
- **GPU验证**: Python + vLLM + PyTorch
- **实验规模**: 44组仿真实验 + 6组GPU验证实验

## 相关工作

| 工作 | 关系 |
|------|------|
| CapKV (arXiv:2604.25975) | IB for KV Cache，我们引用并扩展Phase-aware |
| RDKV (arXiv:2605.08317) | R-D for KV Cache，我们差异化在PD分离+语义distortion |
| KVServe (SIGCOMM 2026) | 系统级加速baseline |
| PDTrim (arXiv:2509.04467) | PD分离带宽优化baseline |
| Mooncake / HiCache | KV分层存储相关工作 |

## 项目状态

- ✅ 仿真框架完整（44组实验）
- ✅ 4-Agent + Scheduler架构验证
- ⏳ GPU真实实验验证（脚本已准备）
- ⏳ 论文撰写

## License

Private Repository - 研究内容保密
