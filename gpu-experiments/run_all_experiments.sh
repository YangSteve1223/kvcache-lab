#!/bin/bash
# =============================================================================
# kvcache-lab GPU实验一键运行脚本
#
# 功能：
#   - 按顺序运行所有GPU实验
#   - 结果保存到 gpu-experiments/results/
#   - 生成汇总报告
#
# 使用方式：
#   bash run_all_experiments.sh [--skip-baseline] [--skip-taa] [--skip-sws] [--skip-eviction]
#   bash run_all_experiments.sh --model-path ./models/my-model --all
#
# 作者：kvcache-lab Team
# =============================================================================

set -e  # 遇到错误立即退出

# =============================================================================
# 颜色输出
# =============================================================================

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

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

log_section() {
    echo ""
    echo -e "${CYAN}============================================================${NC}"
    echo -e "${CYAN}  $1${NC}"
    echo -e "${CYAN}============================================================${NC}"
}

# =============================================================================
# 配置区域
# =============================================================================

# 默认配置
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
MODEL_PATH="${MODEL_PATH:-./models/${MODEL_NAME//\//_}}"

# GPU配置
GPU_ID="${GPU_ID:-0}"
PREFILL_GPU="${PREFILL_GPU:-0}"
DECODE_GPU="${DECODE_GPU:-1}"

# 实验配置
NUM_RUNS="${NUM_RUNS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

# 目录配置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/results"
LOGS_DIR="${SCRIPT_DIR}/logs"

# 创建目录
mkdir -p "$RESULTS_DIR"
mkdir -p "$LOGS_DIR"

# =============================================================================
# 参数解析
# =============================================================================

SKIP_BASELINE=false
SKIP_PD=false
SKIP_TAA=false
SKIP_SWS=false
SKIP_EVICTION=false
RUN_ALL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-baseline)
            SKIP_BASELINE=true
            shift
            ;;
        --skip-pd)
            SKIP_PD=true
            shift
            ;;
        --skip-taa)
            SKIP_TAA=true
            shift
            ;;
        --skip-sws)
            SKIP_SWS=true
            shift
            ;;
        --skip-eviction)
            SKIP_EVICTION=true
            shift
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --num-runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        --all)
            RUN_ALL=true
            shift
            ;;
        --help|-h)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --skip-baseline      跳过基线推理实验"
            echo "  --skip-pd            跳过PD分离实验"
            echo "  --skip-taa           跳过TAA实验"
            echo "  --skip-sws           跳过SWS实验"
            echo "  --skip-eviction      跳过Eviction实验"
            echo "  --all                运行所有实验（默认）"
            echo "  --model-path PATH    模型路径"
            echo "  --model-name NAME    模型名称"
            echo "  --num-runs N         每个实验运行次数"
            echo "  --help, -h           显示此帮助"
            exit 0
            ;;
        *)
            log_error "未知参数: $1"
            exit 1
            ;;
    esac
done

# 如果没有指定跳过任何实验，默认运行全部
if [ "$RUN_ALL" = true ] || \
   { [ "$SKIP_BASELINE" = false ] && [ "$SKIP_PD" = false ] && \
     [ "$SKIP_TAA" = false ] && [ "$SKIP_SWS" = false ] && [ "$SKIP_EVICTION" = false ]; }; then
    RUN_ALL=true
fi

# =============================================================================
# 前置检查
# =============================================================================

log_section "前置检查"

# 检查Python
if ! command -v python3 &> /dev/null; then
    log_error "Python3 未安装"
    exit 1
fi

PYTHON_VER=$(python3 --version 2>&1 | awk '{print $2}')
log_success "Python版本: $PYTHON_VER"

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    log_warning "模型路径不存在: $MODEL_PATH"
    log_info "请先运行 setup.sh 下载模型，或使用 --model-path 指定正确的路径"
    
    # 询问是否继续（仅用于测试某些不需要模型的实验）
    read -p "是否继续运行（某些实验将被跳过）? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    log_success "模型路径: $MODEL_PATH"
fi

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    log_info "GPU信息:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    log_warning "nvidia-smi 未找到，GPU实验可能无法运行"
fi

# =============================================================================
# 环境变量设置
# =============================================================================

export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# =============================================================================
# 辅助函数
# =============================================================================

run_experiment() {
    local name="$1"
    local command="$2"
    local output_file="$3"
    
    log_section "运行实验: $name"
    
    local start_time=$(date +%s)
    
    # 创建日志文件
    local log_file="${LOGS_DIR}/${name}_$(date +%Y%m%d_%H%M%S).log"
    
    # 运行实验
    if eval "$command" 2>&1 | tee "$log_file"; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        
        log_success "$name 完成 (耗时: ${duration}秒)"
        
        # 移动结果文件
        if [ -n "$output_file" ] && [ -f "$output_file" ]; then
            local timestamp=$(date +%Y%m%d_%H%M%S)
            mv "$output_file" "${RESULTS_DIR}/${name}_${timestamp}.json"
            log_info "结果已保存: ${RESULTS_DIR}/${name}_${timestamp}.json"
        fi
        
        return 0
    else
        log_error "$name 失败"
        return 1
    fi
}

# =============================================================================
# 实验1: 基线推理
# =============================================================================

if [ "$SKIP_BASELINE" = false ] && [ -d "$MODEL_PATH" ]; then
    run_experiment \
        "baseline_inference" \
        "cd '$SCRIPT_DIR' && python3 baseline_inference.py \
            --model-path '$MODEL_PATH' \
            --context-length 4096 \
            --num-runs $NUM_RUNS \
            --save-results \
            --output-dir '$RESULTS_DIR'" \
        ""
else
    log_warning "跳过基线推理实验"
fi

# =============================================================================
# 实验2: PD分离
# =============================================================================

if [ "$SKIP_PD" = false ] && [ -d "$MODEL_PATH" ]; then
    # 检查GPU数量
    if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -ge 2 ]; then
        run_experiment \
            "pd_separation" \
            "cd '$SCRIPT_DIR' && python3 pd_separation.py \
                --mode simulate \
                --model-path '$MODEL_PATH' \
                --p-device cuda:${PREFILL_GPU} \
                --d-device cuda:${DECODE_GPU} \
                --bandwidth 10gbps \
                --num-runs $NUM_RUNS \
                --save-results \
                --output-dir '$RESULTS_DIR'" \
            ""
    else
        log_warning "需要2个GPU才能运行PD分离实验，当前跳过"
    fi
else
    log_warning "跳过PD分离实验"
fi

# =============================================================================
# 实验3: TAA (Transmission-Aware Attention)
# =============================================================================

if [ "$SKIP_TAA" = false ] && [ -d "$MODEL_PATH" ]; then
    run_experiment \
        "transmission_aware_attention" \
        "cd '$SCRIPT_DIR' && python3 transmission_aware_attention.py \
            --model-path '$MODEL_PATH' \
            --beta-values 0,0.1,0.3,0.5,1.0 \
            --congestion high \
            --num-runs $NUM_RUNS \
            --save-results \
            --output-dir '$RESULTS_DIR'" \
        ""
else
    log_warning "跳过TAA实验"
fi

# =============================================================================
# 实验4: SWS (Semantic Working Set)
# =============================================================================

if [ "$SKIP_SWS" = false ] && [ -d "$MODEL_PATH" ]; then
    run_experiment \
        "semantic_working_set" \
        "cd '$SCRIPT_DIR' && python3 semantic_working_set.py \
            --model-path '$MODEL_PATH' \
            --working-set-ratios 0.3,0.5,0.7,1.0 \
            --num-runs $NUM_RUNS \
            --save-results \
            --output-dir '$RESULTS_DIR'" \
        ""
else
    log_warning "跳过SWS实验"
fi

# =============================================================================
# 实验5: Predictive Eviction
# =============================================================================

if [ "$SKIP_EVICTION" = false ]; then
    # Eviction实验不需要模型，可以运行
    run_experiment \
        "predictive_eviction" \
        "cd '$SCRIPT_DIR' && python3 predictive_eviction.py \
            --eviction-strategies lru,lfu,predictive \
            --kv-cache-size-gb 40 \
            --workload-size 1000 \
            --workload-type hot_cold \
            --save-results \
            --output-dir '$RESULTS_DIR'" \
        ""
else
    log_warning "跳过Eviction实验"
fi

# =============================================================================
# 生成汇总报告
# =============================================================================

log_section "生成汇总报告"

REPORT_FILE="${RESULTS_DIR}/experiment_summary_$(date +%Y%m%d_%H%M%S).md"

cat > "$REPORT_FILE" << EOF
# kvcache-lab GPU实验汇总报告

生成时间: $(date '+%Y-%m-%d %H:%M:%S')

## 实验配置

- 模型: ${MODEL_NAME}
- 模型路径: ${MODEL_PATH}
- GPU ID: ${GPU_ID}
- 运行次数: ${NUM_RUNS}

## 实验结果

### 1. 基线推理

运行状态: $([ "$SKIP_BASELINE" = false ] && [ -d "$MODEL_PATH" ] && echo "✓ 已运行" || echo "✗ 已跳过")

### 2. PD分离

运行状态: $([ "$SKIP_PD" = false ] && [ -d "$MODEL_PATH" ] && echo "✓ 已运行" || echo "✗ 已跳过")

### 3. TAA (Transmission-Aware Attention)

运行状态: $([ "$SKIP_TAA" = false ] && [ -d "$MODEL_PATH" ] && echo "✓ 已运行" || echo "✗ 已跳过")

### 4. SWS (Semantic Working Set)

运行状态: $([ "$SKIP_SWS" = false ] && [ -d "$MODEL_PATH" ] && echo "✓ 已运行" || echo "✗ 已跳过")

### 5. Predictive Eviction

运行状态: $([ "$SKIP_EVICTION" = false ] && echo "✓ 已运行" || echo "✗ 已跳过")

## 结果文件

所有结果文件保存在: ${RESULTS_DIR}/

## 关键验证目标

1. **TAA验证**: 高拥塞时能否降低延迟而不显著损失质量？
2. **SWS验证**: 只传部分KV能否保持质量？
3. **Eviction验证**: Predictive Eviction命中率是否优于LRU？

## 下一步

1. 查看详细结果文件
2. 分析各实验的JSON输出
3. 对比仿真结果与GPU实测结果
4. 根据发现调整仿真参数

---

报告生成完成
EOF

log_success "汇总报告已保存: $REPORT_FILE"

# =============================================================================
# 完成
# =============================================================================

log_section "所有实验完成"

echo ""
echo "============================================================"
echo "                   实验完成!"
echo "============================================================"
echo ""
echo "结果目录: $RESULTS_DIR"
echo "日志目录: $LOGS_DIR"
echo "汇总报告: $REPORT_FILE"
echo ""
echo "请查看汇总报告了解实验结果详情。"
echo "============================================================"
