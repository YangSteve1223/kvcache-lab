# kvcache-lab GPU实验脚本

Runtime KV Memory OS for PD-Disaggregated LLM Serving - 真实GPU验证实验

## 目录结构

```
gpu-experiments/
├── setup.sh                        # 环境配置脚本
├── baseline_inference.py            # 基线推理脚本
├── pd_separation.py                 # PD分离基线脚本
├── transmission_aware_attention.py   # TAA验证脚本
├── semantic_working_set.py          # SWS验证脚本
├── predictive_eviction.py           # Predictive Eviction验证
├── run_all_experiments.sh          # 一键运行所有实验
├── README.md                        # 本文件
├── results/                         # 实验结果输出目录
│   └── *.json                       # 实验结果文件
├── models/                          # 模型文件目录
│   └── meta-llama_Llama-2-7b-chat-hf/
└── shared/                          # P-D节点共享存储目录
```

## 环境要求

### 硬件要求

- **GPU**: NVIDIA A100 80GB 或 A6000 48GB (推荐)
- **最低**: 单GPU 24GB显存
- **PD分离实验**: 需要 2GPU 或 2台独立GPU服务器
- **网络**: 万兆网络 (推荐) 或 TCP socket 模拟

### 软件要求

- Python 3.10+
- CUDA 11.8+ 或 CUDA 12.1+
- PyTorch 2.1.0+
- transformers 4.38.0+
- vLLM 0.4.0+ (可选，用于加速)

### 模型要求

- **主要**: meta-llama/Llama-2-7b-chat-hf (需要HuggingFace token)
- **备选**: Qwen/Qwen1.5-7B (无需认证)

## 快速开始

### 1. 环境配置

```bash
# 进入实验目录
cd kvcache-lab/gpu-experiments

# 运行环境配置脚本
bash setup.sh --hf-token YOUR_HF_TOKEN

# 或使用备选模型
bash setup.sh --model-name Qwen/Qwen1.5-7B
```

### 2. 单个实验运行

```bash
# 基线推理
python3 baseline_inference.py --model-path ./models/my-model --num-runs 3

# PD分离
python3 pd_separation.py --mode simulate --model-path ./models/my-model --bandwidth 10gbps

# TAA验证
python3 transmission_aware_attention.py --model-path ./models/my-model --beta-values 0,0.1,0.3,0.5,1.0

# SWS验证
python3 semantic_working_set.py --model-path ./models/my-model --working-set-ratios 0.3,0.5,0.7,1.0

# Eviction验证
python3 predictive_eviction.py --eviction-strategies lru,lfu,predictive
```

### 3. 一键运行所有实验

```bash
# 运行所有实验
bash run_all_experiments.sh --model-path ./models/my-model

# 跳过某些实验
bash run_all_experiments.sh --model-path ./models/my-model --skip-taa

# 指定运行次数
bash run_all_experiments.sh --model-path ./models/my-model --num-runs 5
```

## 实验说明

### 1. 基线推理 (baseline_inference.py)

**功能**: 单卡完整推理，无PD分离

**测量指标**:
- TTFT (Time To First Token): 首token生成时间
- TPS (Tokens Per Second): 生成吞吐量
- Perplexity: 困惑度（质量指标）
- GPU内存使用

**参数**:
```bash
python3 baseline_inference.py \
    --model-path ./models/my-model \
    --context-length 4096 \      # 上下文长度
    --max-new-tokens 512 \        # 最大输出token数
    --temperature 0.7 \           # 采样温度
    --num-runs 3 \                # 运行次数
    --save-results                # 保存结果
```

### 2. PD分离 (pd_separation.py)

**功能**: P节点(prefill) → KV传输 → D节点(decode)

**测量指标**:
- Prefill时间
- KV传输时间（支持不同带宽模拟）
- Decode时间
- TTFT和E2E延迟
- 传输数据量

**带宽配置**:
- `1gbps`: 1 GB/s (普通网络)
- `5gbps`: 5 GB/s (较好网络)
- `10gbps`: 10 GB/s (万兆网)
- `nvlink`: 900 GB/s (NVLink)

**参数**:
```bash
python3 pd_separation.py \
    --mode simulate \             # 单机模拟模式
    --model-path ./models/my-model \
    --p-device cuda:0 \            # P节点GPU
    --d-device cuda:1 \            # D节点GPU
    --bandwidth 10gbps \           # 带宽配置
    --num-runs 3
```

### 3. TAA验证 (transmission_aware_attention.py)

**核心创新**: Transmission-Aware Attention

**原理**:
```
modified_score = relevance × exp(-β × cost)
```

- β=0: 无TAA，普通attention
- β>0: 考虑KV访问成本，降低远端KV的attention权重
- 拥塞时自动调整β系数

**β参数扫描**: 0, 0.1, 0.3, 0.5, 1.0

**参数**:
```bash
python3 transmission_aware_attention.py \
    --model-path ./models/my-model \
    --beta-values 0,0.1,0.3,0.5,1.0 \
    --congestion high \            # low/medium/high
    --num-runs 3
```

**验证目标**:
- 高拥塞时降低延迟
- 质量损失在可接受范围内（<5%）

### 4. SWS验证 (semantic_working_set.py)

**核心创新**: Semantic Working Set

**原理**:
1. 分析attention pattern，识别活跃语义区域
2. 只传输Working Set内的KV（而非全部）
3. 区域类型: reasoning_chain, code_context, retrieval_chunk等

**Working Set比例**: 30%, 50%, 70%, 100%

**参数**:
```bash
python3 semantic_working_set.py \
    --model-path ./models/my-model \
    --working-set-ratios 0.3,0.5,0.7,1.0 \
    --task-type math              # math/code/qa/conversation
```

**验证目标**:
- 减少传输数据量
- 保持生成质量

### 5. Eviction验证 (predictive_eviction.py)

**核心创新**: Predictive Eviction

**策略对比**:
- **LRU**: Least Recently Used
- **LFU**: Least Frequently Used
- **Predictive**: 基于reuse距离预测的智能驱逐

**原理**:
- 预测token的重用距离和概率
- 热token → GPU HBM
- 温token → CPU RAM
- 冷token → Remote/Compressed

**参数**:
```bash
python3 predictive_eviction.py \
    --eviction-strategies lru,lfu,predictive \
    --kv-cache-size-gb 40 \       # KV Cache大小
    --workload-size 1000 \         # 工作负载大小
    --workload-type hot_cold       # hot_cold/linear_scan/random/zipfian
```

**验证目标**:
- Predictive vs LRU的命中率提升
- 过早/过晚驱逐减少

## 结果说明

### 结果文件格式

所有实验结果保存为JSON格式：

```json
{
    "timestamp": "2024-01-01T12:00:00",
    "config": {
        "model": "...",
        "experiment_type": "...",
        "parameters": {}
    },
    "statistics": {
        "mean": 0.0,
        "std": 0.0,
        "p50": 0.0,
        "p95": 0.0,
        "p99": 0.0
    },
    "results": [...]
}
```

### 关键验证指标

| 实验 | 核心指标 | 成功标准 |
|------|---------|---------|
| TAA | 延迟降低, 质量损失 | 延迟↓10%, 质量损失<5% |
| SWS | 带宽节省, 质量保留 | 带宽↓30%, 质量>90% |
| Eviction | 命中率 | Predictive > LRU |

## 架构说明

### 单机PD分离

```
[P-Node]              [D-Node]
   │                     │
   │ Prefill             │ Decode
   ▼                     ▼
[GPU 0] ──(KV)──> [Shared] ──(KV)──> [GPU 1]
```

### 分布式PD分离（需要SSH）

```
[P-Node Server]              [D-Node Server]
   Host: p-node.local            Host: d-node.local
   Port: 50051                    Port: 50052
```

启动分布式模式：
```bash
# 终端1: 启动P节点
python3 pd_separation.py --mode p-node --host p-node.local --port 50051

# 终端2: 启动D节点
python3 pd_separation.py --mode d-node --host d-node.local --port 50052
```

## 故障排除

### 1. 模型下载失败

```
错误: HF Token无效或未提供
解决: 访问 https://huggingface.co/settings/tokens 获取token
```

### 2. GPU内存不足

```
错误: CUDA out of memory
解决: 
  - 减少batch size
  - 使用更小的context length
  - 使用半精度(FP16)而非BF16
```

### 3. vLLM导入错误

```
错误: ModuleNotFoundError: No module named 'vllm'
解决: pip install vllm==0.4.0
```

### 4. 多GPU实验失败

```
错误: 需要2个GPU
解决: 确保CUDA_VISIBLE_DEVICES设置正确
```

## 预期结果

### TAA实验预期

| β值 | 延迟变化 | 质量变化 |
|-----|---------|---------|
| 0.0 | Baseline | Baseline |
| 0.1 | -2% | -0.5% |
| 0.3 | -8% | -2% |
| 0.5 | -12% | -4% |
| 1.0 | -18% | -8% |

### SWS实验预期

| 比例 | 带宽节省 | 质量保留 |
|-----|---------|---------|
| 30% | 70% | 75% |
| 50% | 50% | 88% |
| 70% | 30% | 95% |
| 100% | 0% | 100% |

### Eviction实验预期

| 策略 | 命中率 | 过早驱逐 |
|-----|-------|---------|
| LRU | 基准 | 基准 |
| LFU | +5% | -10% |
| Predictive | +15% | -30% |

## 参考

- kvcache-lab项目: https://github.com/your-org/kvcache-lab
- TAA原理: `src/agents/CommunicationAgent.ts`
- SWS原理: `src/agents/SemanticAgent.ts`
- Eviction原理: `src/agents/PlacementAgent.ts`

## 联系方式

如有问题，请提交Issue或联系项目维护者。
