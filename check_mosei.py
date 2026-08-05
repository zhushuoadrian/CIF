import pickle
import numpy as np

# 请将这里的路径替换为您截图中 unaligned_50.pkl 的实际相对路径
# 例如：'./data/MOSEI/unaligned_50.pkl'
file_path = '/home/leng/data/mlzmql/CIF_bestcnow1/data/CMU-MOSEI/unaligned_50.pkl' 

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    print("✅ 成功加载 MOSEI 数据集！")
    # 通常数据集结构是 data['train']['audio'] 或者是 data['audio']['train']
    # 这里我们探测一下维度 (打印出来的最后一个数字就是特征维度)
    print("🎵 音频 (Audio) 维度:", data['train']['audio'].shape)
    print("👁️ 视觉 (Vision) 维度:", data['train']['vision'].shape)
    print("📝 文本 (Text)  维度:", data['train']['text'].shape)
    
except Exception as e:
    print("加载出错，请检查路径或数据结构：", e)