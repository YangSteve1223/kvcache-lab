#!/bin/bash
# =============================================================================
# kvcache-lab GPU实验环境配置脚本
# 
# 功能：
#   1. 安装vLLM（兼容版本）
#   2. 安装Python依赖（torch, transformers, accelerate）
#   3. 下载模型（Llama-2-7B-Chat）
#   4. 验证GPU可用性
#
# 使用方式：
#   bash setup.sh [--model-name MODEL_NAME] [--hf-token HF_TOKEN] [--skip-model-download]
#
# 作者：kvcache-lab Team
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 配置区域
# =============================================================================

# 默认配置
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
HF_TOKEN="${HF_TOKEN:-}"  # 需要从HuggingFace获取
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

# vLLM版本（与torch/transformers兼容的版本）
VLLM_VERSION="0.4.0"
TORCH_VERSION="2.1.0"
TRANSFORMERS_VERSION="4.38.0"

# 实验结果目录
RESULTS_DIR="./results"
MODELS_DIR="./models"
SHARED_DIR="./shared"

# =============================================================================
# 颜色输出
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# =============================================================================
# 参数解析
# =============================================================================

SKIP_MODEL_DOWNLOAD=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --skip-model-download)
            SKIP_MODEL_DOWNLOAD=true
            shift
            ;;
        --help|-h)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model-name NAME        模型名称 (默认: meta-llama/Llama-2-7b-chat-hf)"
            echo "  --hf-token TOKEN         HuggingFace访问令牌"
            echo "  --skip-model-download    跳过模型下载"
            echo "  --help, -h               显示此帮助信息"
            exit 0
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# =============================================================================
# 前置检查
# =============================================================================

check_command() {
    if ! command -v "$1" &> /dev/null; then
        log_error "命令 '$1' 未找到，请先安装"
        return 1
    fi
    return 0
}

log_info "开始环境检查..."

# 检查Python
if ! check_command python3; then
    log_error "请安装Python 3.10+"
    exit 1
fi

PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_VER_MAJOR=$(echo $PYTHON_VER | cut -d. -f1)
PYTHON_VER_MINOR=$(echo $PYTHON_VER | cut -d. -f2)

if [[ "$PYTHON_VER_MAJOR" -lt 3 ]] || [[ "$PYTHON_VER_MAJOR" -eq 3 && "$PYTHON_VER_MINOR" -lt 10 ]]; then
    log_error "需要Python 3.10+，当前版本: $PYTHON_VER"
    exit 1
fi

log_success "Python版本检查通过: $PYTHON_VER"

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    log_info "CUDA环境检测:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
else
    log_warning "nvidia-smi未找到，CUDA可能未正确安装"
fi

# =============================================================================
# 创建虚拟环境（可选）
# =============================================================================

VENV_DIR="./venv"

if [ ! -d "$VENV_DIR" ]; then
    log_info "创建Python虚拟环境..."
    python3 -m venv "$VENV_DIR"
    log_success "虚拟环境已创建: $VENV_DIR"
fi

# 激活虚拟环境
source "$VENV_DIR/bin/activate"

log_info "已激活虚拟环境"
python3 --version

# =============================================================================
# 安装系统依赖
# =============================================================================

log_info "安装系统依赖..."

if command -v apt-get &> /dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq build-essential git curl wget
elif command -v yum &> /dev/null; then
    sudo yum groupinstall -y "Development Tools"
    sudo yum install -y git curl wget
fi

log_success "系统依赖安装完成"

# =============================================================================
# 安装PyTorch
# =============================================================================

log_info "安装PyTorch $TORCH_VERSION (CUDA $CUDA_VERSION)..."

pip install --upgrade pip

pip install \
    torch==${TORCH_VERSION} \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

log_success "PyTorch安装完成"

# 验证PyTorch CUDA
python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# =============================================================================
# 安装transformers和其他依赖
# =============================================================================

log_info "安装transformers和其他Python依赖..."

pip install \
    transformers==${TRANSFORMERS_VERSION} \
    accelerate \
    safetensors \
    numpy \
    scipy \
    scikit-learn \
    pandas \
    matplotlib \
    seaborn \
    huggingface_hub

log_success "Python依赖安装完成"

# =============================================================================
# 安装vLLM
# =============================================================================

log_info "安装vLLM $VLLM_VERSION..."

pip install vllm==${VLLM_VERSION}

log_success "vLLM安装完成"

# 验证vLLM
python3 -c "import vllm; print(f'vLLM版本: {vllm.__version__}')"

# =============================================================================
# 创建目录结构
# =============================================================================

log_info "创建实验目录结构..."

mkdir -p "$RESULTS_DIR"
mkdir -p "$MODELS_DIR"
mkdir -p "$SHARED_DIR"

log_success "目录创建完成:"
echo "  - $RESULTS_DIR/    (实验结果)"
echo "  - $MODELS_DIR/      (模型文件)"
echo "  - $SHARED_DIR/      (共享存储，用于P-D节点KV传输)"

# =============================================================================
# GPU验证
# =============================================================================

log_info "GPU验证..."

python3 << 'EOF'
import torch
import sys

print("=" * 60)
print("GPU环境验证")
print("=" * 60)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"✓ CUDA可用")
    print(f"✓ GPU数量: {device_count}")
    
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  - 计算能力: {props.major}.{props.minor}")
        print(f"  - 总内存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  - 多处理器数量: {props.multi_processor_count}")
        
        # 测试张量创建
        try:
            test_tensor = torch.randn(1024, 1024, device=f'cuda:{i}')
            result = test_tensor @ test_tensor.T
            print(f"  - ✓ CUDA计算测试通过")
        except Exception as e:
            print(f"  - ✗ CUDA计算测试失败: {e}")
            sys.exit(1)
else:
    print("✗ CUDA不可用，请检查CUDA安装")
    sys.exit(1)

print("\n" + "=" * 60)
print("GPU环境验证通过!")
print("=" * 60)
EOF

if [ $? -eq 0 ]; then
    log_success "GPU环境验证通过"
else
    log_error "GPU环境验证失败"
    exit 1
fi

# =============================================================================
# 模型下载
# =============================================================================

if [ "$SKIP_MODEL_DOWNLOAD" = true ]; then
    log_warning "跳过模型下载（--skip-model-download）"
else
    # 检查HuggingFace Token
    if [ -z "$HF_TOKEN" ]; then
        log_warning "未提供HuggingFace Token，无法下载Llama模型"
        log_info "请使用以下方式之一提供Token:"
        echo "  1. 命令行参数: --hf-token YOUR_TOKEN"
        echo "  2. 环境变量: export HF_TOKEN=YOUR_TOKEN"
        echo "  3. 或使用备选模型: --model-name Qwen/Qwen1.5-7B"
        
        # 提供使用备选模型的建议
        log_info "推荐使用不需要认证的模型进行测试:"
        echo "  bash $0 --model-name Qwen/Qwen1.5-7B"
    else
        log_info "下载模型: $MODEL_NAME"
        
        python3 << EOF
from huggingface_hub import snapshot_download
import os

model_name = "${MODEL_NAME}"
model_dir = "${MODELS_DIR}/${model_name.replace('/', '_')}"

print(f"开始下载模型: {model_name}")
print(f"保存到: {model_dir}")

os.environ["HF_TOKEN"] = "${HF_TOKEN}"

try:
    snapshot_download(
        repo_id=model_name,
        local_dir=model_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(f"✓ 模型下载完成: {model_dir}")
except Exception as e:
    print(f"✗ 模型下载失败: {e}")
    print("\n提示：如果是Llama模型，需要:")
    print("  1. 访问 https://huggingface.co/meta-llama/Llama-2-7b-chat-hf")
    print("  2. 同意License")
    print("  3. 获取Access Token")
    print("  4. 使用 --hf-token 参数重新运行")
EOF
    fi
fi

# =============================================================================
# 创建配置文件
# =============================================================================

log_info "创建实验配置文件..."

cat > gpu-experiments-config.sh << 'EOF'
# =============================================================================
# kvcache-lab GPU实验配置文件
# =============================================================================

# 模型配置
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
export MODEL_PATH="${MODELS_DIR}/$(echo $MODEL_NAME | tr '/' '_')"

# GPU配置
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PREFILL_GPU_ID="${PREFILL_GPU_ID:-0}"
export DECODE_GPU_ID="${DECODE_GPU_ID:-1}"

# 节点配置
export P_NODE_HOST="${P_NODE_HOST:-localhost}"
export P_NODE_PORT="${P_NODE_PORT:-50051}"
export D_NODE_HOST="${D_NODE_HOST:-localhost}"
export D_NODE_PORT="${D_NODE_PORT:-50052}"

# 实验参数
export MAX_TOKENS="${MAX_TOKENS:-512}"
export TEMPERATURE="${TEMPERATURE:-0.7}"
export TOP_P="${TOP_P:-0.9}"

# 带宽配置 (模拟不同网络条件)
export BANDWIDTH_GBPS="${BANDWIDTH_GBPS:-10}"  # GB/s

# TAA参数
export TAA_BETA_VALUES="0,0.1,0.3,0.5,1.0"

# SWS参数
export SWS_WORKING_SET_RATIOS="0.3,0.5,0.7,1.0"

# Eviction参数
export EVICTION_STRATEGIES="lru,lfu,predictive"
export KV_CACHE_SIZE_GB="${KV_CACHE_SIZE_GB:-40}"  # 模拟限制的KV Cache大小
EOF

log_success "配置文件已创建: gpu-experiments-config.sh"

# =============================================================================
# 完成
# =============================================================================

echo ""
echo "============================================================"
echo "                   环境配置完成!"
echo "============================================================"
echo ""
echo "下一步："
echo "  1. 配置节点: source gpu-experiments-config.sh"
echo "  2. 运行基线实验: python3 baseline_inference.py"
echo "  3. 运行PD分离: python3 pd_separation.py"
echo "  4. 运行TAA实验: python3 transmission_aware_attention.py"
echo "  5. 运行所有实验: bash run_all_experiments.sh"
echo ""
echo "详细说明请查看: README.md"
echo "============================================================"
