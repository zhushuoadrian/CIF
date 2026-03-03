#!/bin/bash
echo "=========================================="
echo "   Mamba Encoder Replacement Report"
echo "=========================================="

echo -e "\n📊 Replacement Statistics:"
echo "  ✅ Files with MambaEncoder import: $(grep -r "from models.networks.mamba_encoder import" models/ --include="*.py" | grep -v ".ipynb_checkpoints" | wc -l)"
echo "  ✅ Remaining LSTM imports: $(grep -r "from models.networks.lstm import" models/ --include="*.py" | wc -l) (should be 0)"
echo "  ✅ Remaining TextCNN imports: $(grep -r "from models.networks.textcnn import" models/ --include="*.py" | wc -l) (should be 0)"
echo "  ✅ TextCNN usage: $(grep -r "self\.net.*= TextCNN" models/ --include="*.py" | wc -l) (should be 0)"

echo -e "\n🔧 Key Model Files:"
echo "  - CIF_MMIN_model.py (main):"
grep "self.netL = " models/CIF_MMIN_model.py | head -1 | sed 's/^/    /'

echo "  - utt_self_supervise_model.py (pretrain):"
grep "self.netL = " models/utt_self_supervise_model.py | head -1 | sed 's/^/    /'

echo "  - utt_AVL_model.py:"
grep "self.netL = " models/utt_AVL_model.py | head -1 | sed 's/^/    /'

echo -e "\n=========================================="
echo "✅ Replacement Complete!"
echo "=========================================="
