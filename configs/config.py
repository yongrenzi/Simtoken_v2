from email.policy import default
import os

import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

import cv2  # type: ignore

import argparse
import json
import os
from typing import Any, Dict, List

# 数据集结构
file_arch = """
./REFAVS/data
    - /media
    - /gt_mask
    - /metadata.csv
    - /audio_embed
    - /image_embed
"""
# print(f">>> File arch: {file_arch}")

parser = argparse.ArgumentParser(
    description=(
        "SimToken"
    )
)



parser.add_argument("--vision_pretrained",type=str,default='/mnt/HHD_14T/code/lyj/Simtoken_v2/pretrained_model/sam_vit_h_4b8939.pth')
parser.add_argument("--vision_tower",type=str,default='/mnt/HHD_14T/code/Happypk/Crab/pretrained_model/clip-vit-large-patch14')
parser.add_argument("--mllm",type=str,default='/mnt/HHD_14T/code/lyj/Simtoken_v2/pretrained_model/Chat-UniVi-7B-v1.5')

parser.add_argument("--conv_template",type=int,default=1)
parser.add_argument("--ct_weight",type=float,default=0.1)
parser.add_argument("--input_type",type=str,default='refer')
parser.add_argument("--compress",action='store_false',default=False)
parser.add_argument("--start",type=int,default=0)
parser.add_argument("--use_av_attention",action='store_true',default=False, help="Enable audio-visual attention module")


parser.add_argument("--name",type=str,default='testrun_Fvt_F_vs_cluster_remove1')
# path to ref-avs dataset
parser.add_argument("--data_dir",type=str,default='/mnt/HHD_14T/code/Happypk/Crab/REFAVS',help=f"The data paranet dir. File arch should be: {file_arch}")
# path to pretrained checkpoints
parser.add_argument("--saved_model",type=str,default=None, help="the pretrained simtoken pth")


parser.add_argument("--log_root",type=str,default='log', help="where to save log during training")
parser.add_argument("--checkpoint_root",type=str,default='checkpoints', help="where to save trained checkpoints during training")

parser.add_argument("--visualization_root",type=str,default='visualization', help="where to save visualization result during test")

# Resume training from checkpoint
parser.add_argument("--resume", action='store_true', default=False, help="Resume training from checkpoint")
parser.add_argument("--resume_from", type=str, default="/mnt/HHD_14T/code/lyj/Simtoken_v2/checkpoints/testrun_Fvt_F_vs_remove_epoch1_checkpoint.pth", help="Path to checkpoint directory (without _lora/_non_lora suffix)")


# parser.add_argument("--show_params", action='store_true', help=f"Show params names with Requires_grad==True.")

# learning rate
parser.add_argument("--lr", type=float, default=5e-5, help='lr to fine tuning adapters.')
# epochs
parser.add_argument("--epochs", type=int, default=5, help='epochs to fine tuning adapters.')
parser.add_argument("--batch_size", type=int, default=4)


parser.add_argument("--gpu_id", type=str, default="0", help="The GPU device to run generation on.")

parser.add_argument("--run", type=str, default='train', help="train, test")

parser.add_argument("--frame_n", type=int, default=10, help="Frame num of each video. Fixed to 10.")
parser.add_argument("--text_max_len", type=int, default=25, help="Maximum textual reference length.")



args = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
# print(f'>>> Sys: set "CUDA_VISIBLE_DEVICES" - GPU: {args.gpu_id}')
