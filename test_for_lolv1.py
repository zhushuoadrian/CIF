# -*- coding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import AverageMeter
from datasets.LoL_DataLoader import TestData
from pytorch_msssim import ssim
from models import *
import random
import numpy as np
import cv2  # ✅ 新增：用于保存图片

# ============== 固定随机种子函数 ==============
def set_seed(seed=8001):
    """
    固定所有随机种子，确保实验可重复
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# 在最开始就固定种子
set_seed(8001)
# ============================================

# 1. 配置
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='sfhformer_t', type=str, help='model name')
parser.add_argument('--num_workers', default=4, type=int, help='number of workers')
parser.add_argument('--weight', default='/root/lanyun-tmp/the07_bsst_gai5_1_dagai/sfhformer_tlowlight_lol_sfhformer_t_best.pth', type=str, help='weight file')
parser.add_argument('--exp', default='lowlight', type=str, help='experiment setting')
parser.add_argument('--test_data_dir', default='/root/lanyun-tmp/data/LOLdataset/eval15', type=str, help='test data dir')
args = parser.parse_args()

# ✅ 新增：创建保存目录
save_img_root = '/root/lanyun-tmp/newnet/result/lolv1'
os.makedirs(save_img_root, exist_ok=True)

# 2. 加载模型结构与参数
setting_filename = os.path.join('configs', args.exp, args.model + '.json')
if not os.path.exists(setting_filename):
    setting_filename = os.path.join('configs', args.exp, 'default.json')
with open(setting_filename, 'r') as f:
    setting = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network = eval(args.model.replace('-', '_'))()
network = nn.DataParallel(network, device_ids=[0]).to(device)

# 3. 加载权重
state_dict = torch.load(args.weight, map_location=device)
if 'state_dict' in state_dict:
    state_dict = state_dict['state_dict']
network.load_state_dict(state_dict)

# ============== 固定 DataLoader 的随机种子 ==============
def worker_init_fn(worker_id):
    worker_seed = 8001 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
# ====================================================

# 4. 加载测试集
test_dataset = TestData(8, args.test_data_dir)
test_loader = DataLoader(test_dataset,
                         batch_size=1,
                         shuffle=False,
                         num_workers=args.num_workers,
                         pin_memory=True,
                         worker_init_fn=worker_init_fn)

# 5. 测试函数
def test(test_loader, network):
    PSNR_meter = AverageMeter()
    SSIM_meter = AverageMeter()
    network.eval()
    
    img_idx = 0  # ✅ 新增：图片索引
    with torch.no_grad():
        for batch in tqdm(test_loader):
            source_img = batch['source'].cuda()
            target_img = batch['target'].cuda()
            output = network(source_img).clamp_(0, 1)
            mse_loss = F.mse_loss(output, target_img, reduction='none').mean((1, 2, 3))
            psnr = 10 * torch.log10(1 / mse_loss).mean()
            ssim_score = ssim(output, target_img, data_range=1, size_average=False).mean()
            PSNR_meter.update(psnr.item(), source_img.size(0))
            SSIM_meter.update(ssim_score.item(), source_img.size(0))
            
            # ✅ 新增：保存图片
            source_np = (source_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            target_np = (target_img[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            output_np = (output[0].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            
            # 转换颜色空间（RGB→BGR）
            source_bgr = cv2.cvtColor(source_np, cv2.COLOR_RGB2BGR)
            target_bgr = cv2.cvtColor(target_np, cv2.COLOR_RGB2BGR)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            
            # 保存单独图片
            cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}_source.jpg"), source_bgr)
            cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}_target.jpg"), target_bgr)
            cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}_output.jpg"), output_bgr)
            
            # 创建对比图（横向拼接：source | output | target）
            comparison = np.hstack([source_bgr, output_bgr, target_bgr])
            cv2.imwrite(os.path.join(save_img_root, f"{img_idx:04d}_comparison.jpg"), comparison)
            
            img_idx += 1
    
    print("Test PSNR: {:.4f}".format(PSNR_meter.avg))
    print("Test SSIM: {:.4f}".format(SSIM_meter.avg))

# 6. 执行测试
print(f"开始测试，结果将保存到: {save_img_root}")
test(test_loader, network)
print(f"测试完成！图片已保存到: {save_img_root}")