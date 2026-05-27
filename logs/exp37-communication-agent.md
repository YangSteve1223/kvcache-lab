# exp37: Communication Agent 实验日志

## 实验日期
2024

## 实验目的
验证Communication Agent的通信成本计算逻辑和Transmission-Aware Attention效果。

## 实验配置
- Token数量: 32
- 层数: 32
- KV大小: 1024 bytes/token
- GPU内存: 32GB
- 带宽测试: 10Gbps, 50Gbps, 100Gbps, 400Gbps

## 实验结果

### 1. 访问成本基准
| 位置 | 访问成本(ms) |
|------|-------------|
| GPU HBM | ~0.001 |
| CPU RAM | ~0.05 |
| Remote GPU | ~0.5 |
| Compressed | ~1.0 |

### 2. 带宽对远程KV访问成本的影响
| 带宽 | 访问成本(ms) | 传输占比 |
|------|-------------|---------|
| 10Gbps | 0.822 | 39.1% |
| 50Gbps | 0.182 | 39.1% |
| 100Gbps | 0.102 | 39.1% |
| 400Gbps | 0.038 | 39.1% |

### 3. 拥塞级别效果
| 拥塞级别 | β系数 | 带宽利用率影响 |
|---------|-------|---------------|
| low | 0.5 | 轻微 |
| medium | 1.0 | 明显 |
| high | 2.0 | 强烈 |

### 4. Transmission-Aware Attention
在高拥塞时，远程token的attention score显著降低：
- Token 0 (GPU): 原始0.5 → 调整后0.4998 (变化-0.04%)
- Token 2 (Remote): 原始0.1 → 调整后0.086 (变化-14%)
- Token 3 (Compressed): 原始0.05 → 调整后0.035 (变化-30%)

## 结论
✓ Communication Agent成功计算token级别访问成本
✓ 层级别平均成本正确
✓ 拥塞级别检测准确
✓ Transmission-Aware Attention有效降低远程KV的影响
