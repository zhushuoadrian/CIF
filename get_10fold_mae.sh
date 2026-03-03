#!/bin/bash

# ================= 配置区域 =================
# 1. 实验名称 (完整的文件夹名称)
BASE_NAME="CIF_MMIN_MOSI_block_5_run_0_1" 

# 2. 日志根目录
LOG_DIR="./logs"

# 3. 7种模态 (对应 result_total.tsv, result_azz.tsv 等)
MODALITIES=("total" "azz" "zvz" "zzl" "avz" "azl" "zvl")

# 4. 统计的Epoch数量
NUM_EPOCHS=10
# ===========================================

FMT="%-6s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n"
DIVIDER="------------------------------------------------------------------------------------------------"

echo ""
echo "================================================================================================"
echo "实验名称: $BASE_NAME"
echo "搜索路径: $LOG_DIR/$BASE_NAME/results"
echo "指标统计: MAE (前 $NUM_EPOCHS 个 Epoch)"
echo "================================================================================================"

# 检查目录是否存在
TARGET_DIR="$LOG_DIR/$BASE_NAME/results"
if [ ! -d "$TARGET_DIR" ]; then
    echo "❌ 错误: 目录不存在 $TARGET_DIR"
    echo ""
    echo "当前 logs 目录内容:"
    ls -la "$LOG_DIR" | grep -i mosi
    exit 1
fi

echo "✅ 找到结果目录: $TARGET_DIR"
echo ""

# 1. 打印表头
printf "$FMT" "Epoch" "TOTAL" "AZZ" "ZVZ" "ZZL" "AVZ" "AZL" "ZVL"
echo "$DIVIDER"

# 初始化求和数组
declare -A SUMS
declare -A COUNTS
for mod in "${MODALITIES[@]}"; do 
    SUMS[$mod]=0
    COUNTS[$mod]=0
done

# 2. 循环遍历每个 Epoch (1 到 10)
for i in {1..10}; do
    line_output="$i"
    
    # 遍历每个模态
    for mod in "${MODALITIES[@]}"; do
        FILE="$TARGET_DIR/result_${mod}.tsv"
        VAL="-"
        
        if [ -f "$FILE" ]; then
            # 读取第 i+1 行 (第1行是表头,第2行是Epoch 1)
            VAL=$(awk -v row=$((i+1)) 'NR==row && $1 ~ /^[0-9]+(\.[0-9]+)?$/ {print $1}' "$FILE")
        fi
        
        # 检查是否为有效数字
        if [[ "$VAL" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
            SUMS[$mod]=$(awk "BEGIN {print ${SUMS[$mod]} + $VAL}")
            COUNTS[$mod]=$((COUNTS[$mod] + 1))
            VAL=$(printf "%.4f" "$VAL")
        else
            VAL="-"
        fi
        
        line_output="$line_output $VAL"
    done
    
    printf "$FMT" $line_output
done

echo "$DIVIDER"

# 3. 计算平均值
avg_output="AVG"
for mod in "${MODALITIES[@]}"; do
    count=${COUNTS[$mod]}
    sum=${SUMS[$mod]}
    if [ "$count" -gt 0 ]; then
        avg=$(awk "BEGIN {printf \"%.4f\", $sum / $count}")
        avg_output="$avg_output $avg"
    else
        avg_output="$avg_output -"
    fi
done

printf "$FMT" $avg_output
echo "================================================================================================"
echo ""
echo "说明:"
echo "- Epoch 1-10: 每个训练周期的 MAE 值"
echo "- AVG: 前 $NUM_EPOCHS 个 Epoch 的平均 MAE"
echo "- MAE (Mean Absolute Error): 越低越好"
echo ""
echo "模态说明:"
echo "  TOTAL: 完整三模态 (Audio + Visual + Text)"
echo "  AZZ:   只有 Audio (Visual和Text缺失)"
echo "  ZVZ:   只有 Visual (Audio和Text缺失)"
echo "  ZZL:   只有 Text (Audio和Visual缺失)"
echo "  AVZ:   Audio + Visual (Text缺失)"
echo "  AZL:   Audio + Text (Visual缺失)"
echo "  ZVL:   Visual + Text (Audio缺失)"
echo ""