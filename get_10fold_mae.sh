#!/bin/bash

# ================= 配置区域 (已根据您的路径修正) =================
# 1. 实验名称 (匹配文件夹的前缀)
BASE_NAME="name=CIF_MMIN_MOSI_FusionOpt_Mamba_REAL_V1" 

# 2. 日志根目录 (根据您提供的 autodl-tmp/CIF/logs/...)
LOG_DIR="./logs"

# 7种模态 (对应 result_total.tsv, result_azz.tsv 等)
MODALITIES=("total" "azz" "zvz" "zzl" "avz" "azl" "zvl")
# ===========================================

FMT="%-6s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n"
DIVIDER="------------------------------------------------------------------------------------------------"

echo ""
echo "================================================================================================"
echo "实验名称: $BASE_NAME"
echo "搜索路径: $LOG_DIR"
echo "指标统计: MAE (读取 result_*.tsv 文件)"
echo "================================================================================================"

# 1. 打印表头
printf "$FMT" "Fold" "TOTAL" "AZZ" "ZVZ" "ZZL" "AVZ" "AZL" "ZVL"
echo "$DIVIDER"

# 初始化
declare -A SUMS
declare -A COUNTS
for mod in "${MODALITIES[@]}"; do SUMS[$mod]=0; COUNTS[$mod]=0; done

# 2. 循环遍历 1 到 10 折
for i in {1..10}; do
    line_output="$i"
    
    # === 智能查找逻辑 (修正版) ===
    # 1. 在 ./logs 下查找
    # 2. 名字包含 BASE_NAME (Mamba_Fix)
    # 3. 名字包含 fold1 (注意: 这里兼容 fold1, fold01, run_fold1 等写法)
    TARGET_DIR=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*${BASE_NAME}*fold${i}" | sort -r | head -n 1)

    # 如果还是没找到，尝试找带有 _idx1 后缀的 (以防万一)
    if [ -z "$TARGET_DIR" ]; then
        TARGET_DIR=$(find "$LOG_DIR" -maxdepth 1 -type d -name "*${BASE_NAME}*fold${i}_*" | sort -r | head -n 1)
    fi

    for mod in "${MODALITIES[@]}"; do
        VAL="-"
        
        if [ -n "$TARGET_DIR" ]; then
            FILE="$TARGET_DIR/results/result_${mod}.tsv"
            
            if [ -f "$FILE" ]; then
                # 读取最后一行有效的数字
                VAL=$(awk '$1 ~ /^[0-9]+(\.[0-9]+)?$/ { last_val = $1 } END { print last_val }' "$FILE")
            fi
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
echo "提示: 脚本正在搜索 $LOG_DIR 下名为 $BASE_NAME... 的文件夹"
echo ""