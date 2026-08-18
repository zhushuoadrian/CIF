import os
import pickle
import numpy as np

# 配置路径
pkl_path = './data/CMU-MOSEI/unaligned_50.pkl'
target_dir = './data/CMU-MOSEI/target/'

print("正在读取 MOSEI pkl 数据...")
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

# 提取数据的辅助函数
def get_split_data(split_name):
    split_data = data[split_name]
    
    ids = split_data.get('id', [])
    labels = split_data.get('labels', None)
    if labels is None:
        labels = split_data.get('regression_labels', [])
        
    clean_ids = []
    for vid in ids:
        if isinstance(vid, np.ndarray):
            vid = vid[0]
        if isinstance(vid, bytes):
            vid = vid.decode('utf-8')
        clean_ids.append(str(vid))
        
    return np.array(clean_ids), np.array(labels)

# 提取三个官方标准划分
print("正在提取标准 Train / Valid / Test 划分...")
trn_ids, trn_labels = get_split_data('train')
val_ids, val_labels = get_split_data('valid')
tst_ids, tst_labels = get_split_data('test')

# 欺骗 DataLoader：将标准划分复制 10 份，填满 target/1 到 target/10
print("正在生成 .npy 文件并填充到 10 个交叉验证文件夹...")
for i in range(1, 11):
    fold_dir = os.path.join(target_dir, str(i))
    os.makedirs(fold_dir, exist_ok=True)
    
    # 写入训练集 (Train)
    np.save(os.path.join(fold_dir, 'trn_int2name.npy'), trn_ids)
    np.save(os.path.join(fold_dir, 'trn_label.npy'), trn_labels)
    
    # 写入验证集 (Valid)
    np.save(os.path.join(fold_dir, 'val_int2name.npy'), val_ids)
    np.save(os.path.join(fold_dir, 'val_label.npy'), val_labels)
    
    # 写入测试集 (Test)
    np.save(os.path.join(fold_dir, 'tst_int2name.npy'), tst_ids)
    np.save(os.path.join(fold_dir, 'tst_label.npy'), tst_labels)

print(f"✅ 成功！MOSEI 标签拆分完成！存放于: {target_dir}")