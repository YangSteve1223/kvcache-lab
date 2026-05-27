#!/bin/bash
# =============================================================================
# kvcache-lab GPU实验一键部署脚本
# 在AutoDL RTX PRO 6000 96GB上运行
# =============================================================================

set -e

WORK_DIR=/root/autodl-tmp
PROJECT_DIR=$WORK_DIR/kvcache-lab
RESULT_DIR=$WORK_DIR/experiment_results
LOG_FILE=$WORK_DIR/experiment_deploy.log

echo "============================================" | tee -a $LOG_FILE
echo "kvcache-lab GPU实验部署" | tee -a $LOG_FILE
echo "开始时间: $(date)" | tee -a $LOG_FILE
echo "============================================" | tee -a $LOG_FILE

# 1. 安装依赖
echo "[1/5] 安装Python依赖..." | tee -a $LOG_FILE
pip install vllm transformers torch numpy accelerate safetensors -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com 2>&1 | tee -a $LOG_FILE

# 验证安装
python3 -c "import torch; import transformers; import vllm; print(f'PyTorch={torch.__version__}, CUDA={torch.version.cuda}, vLLM={vllm.__version__}')" 2>&1 | tee -a $LOG_FILE

# 2. 下载模型（使用AutoDL镜像加速）
echo "[2/5] 下载模型 Qwen2.5-7B-Instruct..." | tee -a $LOG_FILE
python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print('开始下载tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct', trust_remote_code=True)
print('Tokenizer下载完成')
print('开始下载模型(仅测试能否加载)...')
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-7B-Instruct',
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
print(f'模型加载成功! 显存: {torch.cuda.memory_allocated()/1e9:.1f}GB')
del model
import gc; gc.collect(); torch.cuda.empty_cache()
print('模型已释放，缓存已保存')
" 2>&1 | tee -a $LOG_FILE

# 3. 设置tc带宽限速工具
echo "[3/5] 检查tc限速工具..." | tee -a $LOG_FILE
which tc && echo "tc可用" | tee -a $LOG_FILE || echo "tc不可用，将安装" | tee -a $LOG_FILE
apt-get install -y iproute2 2>/dev/null || true

# 4. 创建结果目录
echo "[4/5] 创建结果目录..." | tee -a $LOG_FILE
mkdir -p $RESULT_DIR

# 5. 运行Phase A实验
echo "[5/5] 开始Phase A实验: G1→G2→G3→G3d..." | tee -a $LOG_FILE
cd $PROJECT_DIR/gpu-experiments

python3 run_all_experiments.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --output-dir $RESULT_DIR \
    --phase A \
    2>&1 | tee -a $LOG_FILE

echo "============================================" | tee -a $LOG_FILE
echo "Phase A实验完成!" | tee -a $LOG_FILE
echo "结束时间: $(date)" | tee -a $LOG_FILE
echo "结果目录: $RESULT_DIR" | tee -a $LOG_FILE
echo "============================================" | tee -a $LOG_FILE

# 列出结果
ls -la $RESULT_DIR/ | tee -a $LOG_FILE
