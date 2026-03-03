import os
import h5py
import numpy as np
import pickle
import sys

# ================= 配置区域 =================
# 1. 这里填你解压后的 MOSI_features_2023 所在的绝对路径
#    (注意：请根据你实际存放的位置修改！)
# ================= 配置区域 (修改为 IEMOCAP) =================
# 1. 源数据路径
#SOURCE_ROOT = '/root/autodl-tmp/CIF/data/IEMOCAP_features_2021' 

# 2. 文件名 (注意：IEMOCAP 的音频文件可能叫 comparE.h5，去文件夹确认一下！)
#AUDIO_FILE = os.path.join(SOURCE_ROOT, 'A', 'comparE.h5')  # <-- 确认文件名！
#VIDEO_FILE = os.path.join(SOURCE_ROOT, 'V', 'denseface.h5')
#TEXT_FILE  = os.path.join(SOURCE_ROOT, 'L', 'bert_large.h5')

# 3. 标签路径 (IEMOCAP 通常也是用 10 折，或者 target/1)
#TARGET_DIR = os.path.join(SOURCE_ROOT, 'target', '1') 

# 4. 输出路径 (生成的 pkl 放在哪里)
# 注意：代码里通常默认找 'data/IEMOCAP_MISA' 或者类似的，建议保持规范
#OUTPUT_DIR = '/root/autodl-tmp/CIF/data/IEMOCAP_MISA'




# ================= 配置区域 (修改为 MSP) =================
# 1. 源数据路径 (你刚才上传解压的文件夹)
SOURCE_ROOT = '/root/autodl-tmp/CIF/data/MSP-IMPROV_features_2021' 

# 2. 文件名 
# ⚠️注意：去 MSP 的文件夹里看一眼，确认音频文件是叫 acoustic.h5 还是 comparE.h5
# 通常 MSP 和 IEMOCAP 类似，可能叫 comparE.h5，也可能叫 acoustic.h5
AUDIO_FILE = os.path.join(SOURCE_ROOT, 'A', 'comparE_raw.h5') # <--- 请务必确认文件名！
VIDEO_FILE = os.path.join(SOURCE_ROOT, 'V', 'denseface.h5')
TEXT_FILE  = os.path.join(SOURCE_ROOT, 'L', 'bert_large.h5')

# 3. 标签路径 (MSP 通常是 12折，或者看 target 文件夹里是 1 还是 10)
# 建议先填 'target/1' 试一下
TARGET_DIR = os.path.join(SOURCE_ROOT, 'target', '1') 

# 4. 输出路径
OUTPUT_DIR = '/root/autodl-tmp/CIF/data/MSP_MISA'
# ==========================================================






# =====================配置区域 (修改为 MOSI)=================
#SOURCE_ROOT = '/root/autodl-tmp/CIF/data/MOSI_features_2023' 

# 2. h5 文件所在的子目录名 (根据你刚才说的 A, V, L 文件夹)
#    假设 A 文件夹里是 acoustic.h5，如果文件名不同请修改
#AUDIO_FILE = os.path.join(SOURCE_ROOT, 'A', 'acoustic.h5')  
#VIDEO_FILE = os.path.join(SOURCE_ROOT, 'V', 'visual.h5')
#TEXT_FILE  = os.path.join(SOURCE_ROOT, 'L', 'bert_large.h5') # 你之前截图里是 bert_large.h5

# 3. 使用哪一折的数据 (通常用 1 或 10)
#TARGET_DIR = os.path.join(SOURCE_ROOT, 'target', '10') 

# 4. 输出结果存放的路径 (你的代码报错找的地方)
#OUTPUT_DIR = '/root/autodl-tmp/CIF/data/mosi_MISA'
# ===========================================






def check_files():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    
    files = [AUDIO_FILE, VIDEO_FILE, TEXT_FILE]
    for f in files:
        if not os.path.exists(f):
            print(f"❌ 错误: 找不到文件 {f}")
            print("请检查 '配置区域' 的路径是否正确！")
            print("提示：你可能需要把 .h5 文件从 A/V/L 文件夹里拿出来，或者修改代码里的路径。")
            sys.exit(1)
    print("✅ 原始文件路径检查通过")

def get_data(args, dataset_name):
    # 读取 h5 文件
    # 注意：如果你的 h5 不在 A/V/L 文件夹里，而是在根目录，请修改上面的路径
    try:
        audio_h5 = h5py.File(AUDIO_FILE, 'r')
        video_h5 = h5py.File(VIDEO_FILE, 'r')
        text_h5  = h5py.File(TEXT_FILE, 'r')
    except Exception as e:
        print(f"读取h5文件失败: {e}")
        sys.exit(1)

    # 读取 npy 清单
    # dataset_name 是 'trn' (train), 'val' (dev), 或 'tst' (test)
    ids_path = os.path.join(TARGET_DIR, f"{dataset_name}_int2name.npy")
    label_path = os.path.join(TARGET_DIR, f"{dataset_name}_label.npy")
    
    if not os.path.exists(ids_path):
        print(f"❌ 找不到清单文件: {ids_path}")
        sys.exit(1)

    # 加载 ID 和 标签
    ids = np.load(ids_path)
    labels = np.load(label_path)
    
    data_list = []
    
    print(f"正在处理 {dataset_name} 集，共 {len(ids)} 条数据...")

    for i, vid_id in enumerate(ids):
       # ================== 修改开始 ==================
        # 1. 如果 vid_id 是 numpy 数组，先取出来
        if isinstance(vid_id, np.ndarray):
            if vid_id.size == 1:
                vid_id = vid_id.item()
            else:
                vid_id = vid_id[0]
        
        # 2. 如果是 bytes 类型，解码成字符串
        if isinstance(vid_id, bytes):
            vid_id = vid_id.decode('utf-8')
            
        # 3. 强转为字符串 (双重保险)
        vid_id = str(vid_id).strip()
        # ================== 修改结束 ==================
        try:
            # 从 h5 中提取特征
            # 注意：这里假设 h5 的结构是 h5[video_id]['features'] 或者直接 h5[video_id]
            # 根据常见数据集格式尝试读取
            
            # 音频
            audio = np.array(audio_h5[vid_id])
            # 视觉
            video = np.array(video_h5[vid_id])
            # 文本
            text = np.array(text_h5[vid_id])
            
            # 组装字典
            # ================= 修改开始 =================
            # 你的 test.py 期望的是 ((audio, visual, text), label, id) 这种结构
            # 注意：如果不确定顺序，通常是 (text, visual, audio) 或 (audio, visual, text)
            # 这里我们按照 data[0][0][0] 这种深层结构，推测它是把特征包了一层
            
            # 格式：((Text, Visual, Audio), Label, ID)
            # 这是学术界 MISA/MMIN 等代码最常用的格式
            sample = ((text, video, audio), labels[i], vid_id)
            
            data_list.append(sample)
            # ================= 修改结束 =================
        except KeyError:
            print(f"⚠️ 警告: ID {vid_id} 在 h5 文件中未找到，跳过。")
            continue

    # 保存为 pkl
    # 映射名字: trn->train.pkl, val->dev.pkl, tst->test.pkl
    save_name = ""
    if dataset_name == 'trn': save_name = 'train.pkl'
    elif dataset_name == 'val': save_name = 'dev.pkl'
    elif dataset_name == 'tst': save_name = 'test.pkl'
    
    save_path = os.path.join(OUTPUT_DIR, save_name)
    with open(save_path, 'wb') as f:
        pickle.dump(data_list, f)
    
    print(f"✅ 已生成: {save_path} (样本数: {len(data_list)})")

if __name__ == '__main__':
    check_files()
    get_data(None, 'trn')
    get_data(None, 'val') # 生成 dev.pkl
    get_data(None, 'tst')
    print("\n🎉 全部处理完成！现在可以运行 test.py 了。")