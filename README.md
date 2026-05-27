# kvcache-lab

KV Cache压缩仿真实验框架 —— PD分离场景下任务自适应的KV Cache压缩研究

## 研究方向
- PD-Aware KV Compression：P端/D端差异化压缩策略
- Task-Aware Layer Budget Allocation：任务感知层预算分配

## 项目结构
```
src/
├── core/          # 仿真引擎（PDSimulator, KVCacheManager, QualityModel）
├── compression/   # 压缩策略（None/Uniform/PD-Aware/Task-Aware）
└── task/          # 任务感知（TaskClassifier, LayerBudgetAllocator, Profiles）
experiments/       # 实验脚本（exp1-5）
logs/              # 实验日志
tests/             # 测试文件
```

## 快速开始
```bash
npm install
npm test
npx tsx experiments/exp4-full-comparison.ts
```

## 实验结果
| 策略 | TTFT(ms) | E2E(ms) | 压缩比 | 质量 |
|------|----------|---------|--------|------|
| None | 473 | 2061 | 1.00 | 1.00 |
| Uniform | 473 | 2034 | 0.50 | 0.91 |
| PD-Aware | 465 | 1999 | 0.51 | 0.89 |
| Task-Aware | 484 | 2047 | 0.58 | 0.90 |

## 技术栈
- TypeScript + Node.js 22
- DeepSeek API（任务分类验证）
- 零外部运行时依赖（仿真纯TypeScript）

## License
MIT
