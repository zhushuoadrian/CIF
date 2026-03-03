#!/bin/bash
echo "=========================================="
echo "1. models/our/CIF_MMIN_model.py"
echo "=========================================="
if [ -f models/our/CIF_MMIN_model.py ]; then
    cat models/our/CIF_MMIN_model.py
else
    echo "文件不存在"
fi

echo -e "\n=========================================="
echo "2. models/CIF_MMIN_model.py"
echo "=========================================="
if [ -f models/CIF_MMIN_model.py ]; then
    cat models/CIF_MMIN_model.py
else
    echo "文件不存在"
fi

echo -e "\n=========================================="
echo "3. models/utt_self_supervise_model.py (前200行)"
echo "=========================================="
head -200 models/utt_self_supervise_model.py

echo -e "\n=========================================="
echo "4. scripts/our/MOSI_CIF_MMIN.sh"
echo "=========================================="
cat scripts/our/MOSI_CIF_MMIN.sh

echo -e "\n=========================================="
echo "5. scripts/MOSI_utt_self_supervise.sh"
echo "=========================================="
cat scripts/MOSI_utt_self_supervise.sh
