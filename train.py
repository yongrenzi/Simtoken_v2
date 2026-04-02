import transformers
from datasets import REFAVS
from configs import args
from torch.utils.data import DataLoader
from functools import partial
from models.llava import conversation as conversation_lib
# from  models.avs_model import VISAForCausalLM
from  models.avs_model import Simtoken_ForCausalLM
import os
import torch
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model
from torch import optim
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils import utility
import random
import numpy as np
import re
import time
import os
import debugpy

import warnings
warnings.filterwarnings("ignore")

from transformers import logging
logging.set_verbosity_error()


IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
DEFAULT_VIDEO_TOKEN = "<video>"

AUDIO_TOKEN_INDEX = -300
DEFAULT_AUDIO_TOKEN = "<audio>"

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
                isinstance(input_dict[k], list)
                and len(input_dict[k]) > 0
                and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict

def tokenizer_image_audio_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, audio_token_index=AUDIO_TOKEN_INDEX, num_frames=10, return_tensors=None):

    prompt_chunks = re.split(r'(<image>|<audio>|<video>)', prompt)


    prompt_chunks = [chunk for chunk in prompt_chunks if chunk]

    text_chunks = []
    token_types = []
    for chunk in prompt_chunks:
        if chunk == "<image>":
            token_types.append("image")
        elif chunk == "<audio>":
            token_types.append("audio")
        elif chunk == "<video>":
            token_types.append("video")
        else:
            text_chunks.append(chunk)

    tokenized_chunks = [tokenizer(chunk).input_ids for chunk in text_chunks]

    def insert_separators(text_chunks, tokenized_chunks, token_types, image_token_index, audio_token_index, num_frames):
        input_ids = []
        offset = 0
        if (
                len(tokenized_chunks) > 0
                and len(tokenized_chunks[0]) > 0
                and tokenized_chunks[0][0] == tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(tokenized_chunks[0][0])

        min_length = min(len(text_chunks), len(token_types))
        for i in range(min_length):

            input_ids.extend(tokenized_chunks[i][offset:])

            if token_types[i] == "image":
                input_ids.append(image_token_index)
            elif token_types[i] == "audio":
                input_ids.append(audio_token_index)
            elif token_types[i] == "video":
                input_ids.extend([image_token_index] * num_frames)


        if len(text_chunks) > min_length:
            input_ids.extend(tokenized_chunks[min_length][offset:])

        return input_ids

    input_ids = insert_separators(text_chunks, tokenized_chunks, token_types, image_token_index, audio_token_index, num_frames)

    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")
    return input_ids

def collate_fn(batch, tokenizer=None):
    vids = []
    images = []
    image_clips = []
    masks = []
    conversations = []
    audio_feats = []
    image_feats = []
    resizes = []
    orgsizes = []
    refs = []
    refs_num = []
    fids = []


    for data in batch:
        vids.append(data['vid'])
        images.append(data['image'])
        image_clips.append(data['img_clip'])
        masks.append(data['mask'])
        conversations.append(data['conversation'])
        audio_feats.append(data['feat_aud'])
        resizes.append(data['resize'])
        orgsizes.append(data['orgsize'])
        image_feats.append(data['feat_sam'])
        refs_num.append(len(data['ref']))
        fids.append(data['fids'])

        refs.append(data['ref'][0])


    # input_ids = [tokenizer_image_token(conv, tokenizer, return_tensors="pt") for conv in conversations]
    input_ids = [tokenizer_image_audio_token(conv, tokenizer, return_tensors="pt") for conv in conversations]  # list
    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks = input_ids.ne(tokenizer.pad_token_id)

    ref_ids = [tokenizer_image_audio_token(ref, tokenizer, return_tensors="pt") for ref in refs]

    conv = conversation_lib.default_conversation.copy()
    labels = input_ids.clone()

    # sep = conv.sep + conv.roles[1] + ": " # “###Assistant：”
    sep = 'Sure, it is [SEG]'

    for conversation, target in zip(conversations, labels):

        parts = conversation.split(sep)
        # print(parts)

        cur_len = 1
        target[:cur_len] = IGNORE_INDEX

        sep_len = len(tokenizer_image_audio_token(sep, tokenizer)) - 1


        for i in range(len(parts)-1):
            part_len = len(tokenizer_image_audio_token(parts[i], tokenizer)) - 2
            target[cur_len: cur_len + part_len] = IGNORE_INDEX
            cur_len += part_len + sep_len

        target[cur_len:] = IGNORE_INDEX


    return {"vids": vids,
            "images": images,  # list[B]:[T, 3, 1024, 1024]
            "images_clip": image_clips,  # list[B]:[T, 3, 224, 224]
            "masks": masks,  # list[B]:[num_ref, T, H, W]
            "convs": conversations,  # list[B]: str
            "input_ids": input_ids,  # list[B]:[max_len]
            "attention_masks": attention_masks,  # list[B]:[max_len]
            "labels": labels,  # list[B]:[max_len]
            "audio_feats": audio_feats,  # list[B]:[10, 128]
            "resizes": resizes,  # list[B]
            "orgsizes": orgsizes,  # list[B]
            "image_feats": image_feats,
            "ref_ids": ref_ids,  # list[B]: [ref_id_len]
            "refs_num": refs_num,
            "fids": fids
    }


import torch.multiprocessing as mp
if __name__ == "__main__":
    # # # 5678 是监听端口，可以随意改，但要和 launch.json 对应
    # debugpy.listen(("localhost", 1234))
    # print("等待调试器连接...")
    # debugpy.wait_for_client()  # 程序会暂停在这里，直到你按下 F5

    # # 下面是你原本的代码
    # print("调试器已连接，开始执行！")
    mp.set_start_method("spawn")
    set_seed(42)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.mllm,
        cache_dir=None,
        model_max_length=2048,  # 2048
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_tokens = tokenizer.add_tokens("[SEG]")
    seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]  # 32000
    print("seg_token_idx: ", seg_token_idx)

    train_dataset = REFAVS('train', args, tokenizer, input_type='refer')
    val_dataset_s_refer = REFAVS('test_s', args, tokenizer, input_type='refer')
    val_dataset_u_refer = REFAVS('test_u', args, tokenizer, input_type='refer')
    val_dataset_n_refer = REFAVS('test_n', args, tokenizer, input_type='refer')


    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8, worker_init_fn=seed_worker,collate_fn=partial(collate_fn, tokenizer=tokenizer), generator=g)

    val_dataloader_s_refer = DataLoader(val_dataset_s_refer, batch_size=4, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_u_refer = DataLoader(val_dataset_u_refer, batch_size=4, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))
    val_dataloader_n_refer = DataLoader(val_dataset_n_refer, batch_size=4, shuffle=False, num_workers=0, collate_fn=partial(collate_fn, tokenizer=tokenizer))


    model_args = {
        "train_mask_decoder": True,
        "out_dim": 256,  # 256
        "ce_loss_weight": 1.0,
        "dice_loss_weight": 0.5,
        "bce_loss_weight": 2.0,
        "seg_token_idx": seg_token_idx,
        "vision_pretrained": args.vision_pretrained,  # sam_vit_h_xxx.pth
        "vision_tower": args.vision_tower,
        "use_im_start_end": False,
        "compress": args.compress,
        "start": args.start,
        "use_av_attention": args.use_av_attention,  # 添加音视频注意力开关
    }

    # model = Simtoken_ForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.float32, low_cpu_mem_usage=True, **model_args)
    model = Simtoken_ForCausalLM.from_pretrained(args.mllm, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, **model_args)
    print("\nmodel loaded")

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    # vision_tower.to(dtype=torch.float32, device="cuda")
    vision_tower.to(dtype=torch.bfloat16, device="cuda")
    model_args_from_pt = AutoConfig.from_pretrained(args.mllm)
    model_args_from_pt.use_cluster = True
    model_args_from_pt.freeze = False
    model_args_from_pt.mm_tune = True
    model_args_from_pt.spatial_cluster_rate0 = 64
    model_args_from_pt.spatial_cluster_rate1 = 32
    model_args_from_pt.spatial_cluster_rate2 = 16
    model_args_from_pt.temporal_cluster_rate = 0.0625
    model_args_from_pt.use_cluster = True
    model_args_from_pt.vision_tune = False
    model.get_model().initialize_cluster_modules(model_args_from_pt)

    model.get_model().initialize_lisa_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False
    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False

    lora_r = 8
    target_modules = "q_proj,v_proj"
    if lora_r > 0:

        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()

            for name, module in model.named_modules():
                if (

                        isinstance(module, cls)

                        and all(
                    [
                        x not in name
                        for x in [
                        "visual_model",
                        "vision_tower",
                        "mm_projector",
                        "text_hidden_fcs",
                        "audio_feature_layer",
                    ]
                    ]
                )

                        and any([x in name for x in lora_target_modules])
                ):

                    lora_module_names.add(name)
            return sorted(list(lora_module_names))


        lora_alpha = 16
        lora_dropout = 0.05

        lora_target_modules = find_linear_layers(
            model, target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, lora_config)
        print("\nLora deployed")

        model.print_trainable_parameters()

    model = model.to("cuda")
    # Convert entire model to bfloat16 (including LoRA adapters),增加下面这一行
    model = model.to(torch.bfloat16)
    model.resize_token_embeddings(len(tokenizer))


    for name, param in model.audio_feature_layer.named_parameters():
        param.requires_grad = True
        # print(name, param.requires_grad)

    # ===== 设置音视频注意力模块为可训练，并强制保持 FP32 =====
    def reinit_av_attention(av_module):
        """重新初始化 AV Attention 模块的权重"""
        av_module = av_module.float()

        print("[INFO] Re-initializing AV Attention weights after dtype conversion...")
        for name, module in av_module.named_modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, torch.nn.LayerNorm):
                torch.nn.init.constant_(module.weight, 1.0)
                torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, torch.nn.Conv2d):
                torch.nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, torch.nn.MultiheadAttention):
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    torch.nn.init.xavier_uniform_(module.in_proj_weight, gain=1.0)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    torch.nn.init.constant_(module.in_proj_bias, 0.0)

        # 设置所有参数为可训练
        for param in av_module.parameters():
            param.requires_grad = True

        print("av_attention module set to trainable (FP32)")

        # 验证初始化
        print("\n[DEBUG] AV Attention Module Weights After Re-initialization:")
        for name, param in av_module.named_parameters():
            has_nan = torch.isnan(param).any()
            if has_nan:
                print(f"  {name}: HAS NaN! ❌")
            else:
                print(f"  {name}: requires_grad={param.requires_grad}, min={param.min():.4f}, max={param.max():.4f}")
        print()

        return av_module

    # 应用到模型
    if hasattr(model, 'av_attention') and model.av_attention is not None:
        model.av_attention = reinit_av_attention(model.av_attention)
    elif hasattr(model, 'model') and hasattr(model.model, 'av_attention') and model.model.av_attention is not None:
        model.model.av_attention = reinit_av_attention(model.model.av_attention)


    for n, p in model.named_parameters():
        if any(
                [
                    x in n
                    for x in ["lm_head", "embed_tokens", "mask_decoder", "text_hidden_fcs"]
                ]
        ):
            p.requires_grad = True


    print("will save train model")
    # ===== Helper Functions for Checkpoint Management =====
    def save_checkpoint(epoch, model, optimizer, scheduler, save_path):
        """Save complete checkpoint including model, optimizer, and scheduler states"""
        # 1. Save LoRA adapters (备份, 方便后续 merge)
        lora_path = f"{save_path}_epoch{epoch}_lora"
        model.save_pretrained(lora_path)

        # 2. 收集所有可训练参数 (LoRA + 非LoRA), 直接从 state_dict 匹配
        full_state = model.state_dict()
        param_names_grad = {n for n, p in model.named_parameters() if p.requires_grad}

        # state_dict 的 key 可能与 named_parameters 的 key 不完全一致
        # 同时用两种方式匹配, 确保不遗漏
        trainable_state = {}
        for k, v in full_state.items():
            if k in param_names_grad:
                trainable_state[k] = v.cpu()
            else:
                # 检查去掉常见前缀后是否匹配 (peft 包装可能加前缀)
                for pn in param_names_grad:
                    if k.endswith(pn) or pn.endswith(k):
                        trainable_state[k] = v.cpu()
                        break

        print(f"  Trainable params to save: {len(trainable_state)} "
              f"(from {len(param_names_grad)} requires_grad params)")

        # 3. Save complete checkpoint with training state
        checkpoint = {
            'epoch': epoch,
            'trainable_state_dict': trainable_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }

        checkpoint_path = f"{save_path}_epoch{epoch}_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        print(f"Checkpoint saved: epoch {epoch}")
        print(f"  - LoRA backup: {lora_path}")
        print(f"  - Checkpoint: {checkpoint_path}")
        return lora_path, checkpoint_path


    def load_checkpoint(model, optimizer, scheduler, resume_from):
        """Load checkpoint and resume training state"""
        import glob

        # Find the latest checkpoint if resume_from is a directory
        if os.path.isdir(resume_from):
            checkpoint_files = glob.glob(os.path.join(resume_from, "*_checkpoint.pth"))
            if not checkpoint_files:
                print(f"No checkpoint found in {resume_from}")
                return 0
            checkpoint_path = max(checkpoint_files, key=os.path.getctime)
        else:
            checkpoint_path = resume_from

        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return 0

        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')

        # Extract epoch number from checkpoint
        start_epoch = checkpoint['epoch'] + 1

        # 兼容旧格式 (non_lora_state_dict) 和新格式 (trainable_state_dict)
        if 'trainable_state_dict' in checkpoint:
            saved_state = checkpoint['trainable_state_dict']
        elif 'non_lora_state_dict' in checkpoint:
            saved_state = checkpoint['non_lora_state_dict']
            print("[WARN] Using legacy checkpoint format (non_lora_state_dict), LoRA weights may not be included")
        else:
            print("[ERROR] Checkpoint has no recognized state_dict key")
            return 0

        # 加载可训练参数, 并验证实际匹配了多少 key
        model_state = model.state_dict()
        loaded_keys = []
        missing_keys = []
        for k, v in saved_state.items():
            if k in model_state:
                if model_state[k].shape == v.shape:
                    model_state[k] = v
                    loaded_keys.append(k)
                else:
                    print(f"  [WARN] Shape mismatch for {k}: "
                          f"model={model_state[k].shape}, ckpt={v.shape}, skipped")
                    missing_keys.append(k)
            else:
                missing_keys.append(k)

        model.load_state_dict(model_state, strict=False)
        print(f"  Loaded {len(loaded_keys)}/{len(saved_state)} trainable params")
        if missing_keys:
            print(f"  [WARN] {len(missing_keys)} keys in checkpoint not found in model:")
            for mk in missing_keys[:10]:
                print(f"    - {mk}")
            if len(missing_keys) > 10:
                print(f"    ... and {len(missing_keys) - 10} more")

        # 检查模型中 requires_grad 但未被恢复的参数
        restored_set = set(loaded_keys)
        not_restored = [n for n, p in model.named_parameters()
                        if p.requires_grad and n not in restored_set]
        if not_restored:
            print(f"  [WARN] {len(not_restored)} trainable params NOT restored from checkpoint:")
            for nr in not_restored[:10]:
                print(f"    - {nr}")
            if len(not_restored) > 10:
                print(f"    ... and {len(not_restored) - 10} more")

        # Load optimizer state (已在 cuda 上, 无 device mismatch)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"  Loaded optimizer state")

        # Load scheduler state
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"  Loaded scheduler state")

        print(f"Resuming from epoch {start_epoch}")
        return start_epoch

    def valuate(model, dataloader, args, name):
        model.eval()

        total_iou = 0
        total_fscore = 0
        count = 0

        for batch in tqdm(dataloader, desc=f"Evaluating on {name}"):
            input_dict = dict_to_cuda(batch)
            with torch.no_grad():
                output_dict = model.forward(images=input_dict["images"],
                                            images_clip=input_dict["images_clip"],
                                            audio_features=input_dict["audio_feats"],
                                            image_features=input_dict["image_feats"],
                                            input_ids=input_dict["input_ids"],
                                            labels=input_dict["labels"],
                                            attention_masks=input_dict["attention_masks"],
                                            masks_list=input_dict["masks"],
                                            resize_list=input_dict["resizes"],
                                            orgsize_list=input_dict["orgsizes"],
                                            conversation_list=input_dict["convs"],
                                            refs_num=input_dict["refs_num"],
                                            fids=input_dict["fids"],
                                            vids=input_dict["vids"],
                                            contrast=args.ct_weight,
                                            ref_ids=input_dict["ref_ids"],
                                            inference=True)
            pred_masks = output_dict["pred_masks"]  # list[B]:[num_seg, T, H, W]
            gt_masks = output_dict["gt_masks"]  # list[B]:[num_seg, T, H, W]
            for i in range(len(pred_masks)):
                num_seg = pred_masks[i].shape[0]
                T = pred_masks[i].shape[1]
                iou = utility.mask_iou(pred_masks[i], gt_masks[i])
                fscore = utility.Eval_Fmeasure(pred_masks[i], gt_masks[i], None)

                total_iou += iou * num_seg * T
                total_fscore += fscore * num_seg * T
                count += num_seg * T

        print(f"\n  valuate on {name}:  miou: {total_iou/count}  fscore: {total_fscore/count}")

        with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
            f.write(f"valuate on {name}:  miou {total_iou/count}  true fscore {total_fscore/count} \n")


    # ---------------train------------------------------------------

    model.train()
    epochs = args.epochs
    print("init lr:", args.lr)
    optimizer = AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.01)

    gradient_accumulation_steps = int(16 // args.batch_size)
    step_per_epoch = len(train_dataloader) // gradient_accumulation_steps
    total_steps = epochs * step_per_epoch
    warmup_steps = int(total_steps * 0.1)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    # ===== Load checkpoint if resuming =====
    start_epoch = 0
    if args.resume:
        if args.resume_from:
            start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume_from)
        else:
            # Try to load from default checkpoint location
            default_checkpoint = os.path.join(args.checkpoint_root, args.name)
            if os.path.exists(args.checkpoint_root):
                start_epoch = load_checkpoint(model, optimizer, scheduler, default_checkpoint)

        if start_epoch > 0:
            print(f"Successfully resumed from epoch {start_epoch}")
        else:
            print("No checkpoint found, starting from scratch")

    for epoch in range(start_epoch, epochs):

        model.train()
        optimizer.zero_grad()
        running_loss = 0.0

        loop = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(loop):
            input_dict = dict_to_cuda(batch)
            output_dict = model.forward(images=input_dict["images"],
                                        images_clip=input_dict["images_clip"],
                                        audio_features=input_dict["audio_feats"],
                                        image_features=input_dict["image_feats"],
                                        input_ids=input_dict["input_ids"],
                                        labels=input_dict["labels"],
                                        attention_masks=input_dict["attention_masks"],
                                        masks_list=input_dict["masks"],
                                        resize_list=input_dict["resizes"],
                                        orgsize_list=input_dict["orgsizes"],
                                        conversation_list=input_dict["convs"],
                                        refs_num=input_dict["refs_num"],
                                        fids=input_dict["fids"],
                                        vids=input_dict["vids"],
                                        contrast=args.ct_weight,
                                        ref_ids=input_dict["ref_ids"],
                                        epoch=epoch,
                                        inference=False)

            loss = output_dict["loss"]
            loss = loss / gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()


            if (step + 1) % gradient_accumulation_steps == 0:
                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_lr = scheduler.get_lr()[0]
                loop.set_postfix(lr=current_lr, loss=running_loss / ((step + 1) / gradient_accumulation_steps))

        print(f"  Epoch {epoch + 1}, Loss:{running_loss / ((step + 1) / gradient_accumulation_steps) :.4f}, Learning Rate:{scheduler.get_last_lr()[0]:.6f}")


        with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
            f.write(f"Epoch {epoch}: running_loss {running_loss / len(train_dataloader) * gradient_accumulation_steps}  Learning Rate:{scheduler.get_last_lr()[0]:.6f}\n")

        # ===== 每个 epoch 结束后在 seen 数据集上评估 =====
        print(f"\n{'='*60}")
        print(f"Evaluating on test_s (seen) after Epoch {epoch + 1}")
        print(f"{'='*60}")
        # valuate(model, val_dataloader_s_refer, args, 'test_s_refer')
        print(f"{'='*60}\n")

        # 重新设置为训练模式
        # model.train()

        # ===== Save checkpoint after each epoch =====
        checkpoint_base_path = os.path.join(args.checkpoint_root, args.name)
        save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_base_path)

        # ===== Save Final Model: LoRA + Other Trainable Parameters =====
    print("\n" + "="*50)
    print("Training completed! Saving final model...")
    print("="*50)

    # 1. Save LoRA adapters
    lora_save_path = os.path.join(args.checkpoint_root, f"{args.name}_final_lora")
    model.save_pretrained(lora_save_path)
    print(f"Final LoRA adapters saved to {lora_save_path}")

    # 2. Save other trainable parameters (non-LoRA)
    # Collect all trainable parameter names
    trainable_param_names = {n for n, p in model.named_parameters() if p.requires_grad}

    # Filter out LoRA parameters (they contain 'lora' in their names)
    non_lora_trainable = {
        k: v for k, v in model.state_dict().items()
        if k in trainable_param_names and 'lora' not in k.lower()
    }

    non_lora_save_path = os.path.join(args.checkpoint_root, f"{args.name}_final_non_lora.pth")
    torch.save(non_lora_trainable, non_lora_save_path)
    print(f"Final non-LoRA trainable parameters saved to {non_lora_save_path}")
    print(f"Total trainable modules saved: {len(non_lora_trainable)} parameters")
    print("="*50 + "\n")

    # ---------------test on seen & unseen ------------------------------------------
    model.eval()

    valuate(model, val_dataloader_s_refer, args, 'test_s_refer')
    valuate(model, val_dataloader_u_refer, args, 'test_u_refer')

    # ---------------test on Null ------------------------------------------
    model.eval()

    total_metric = 0
    count = 0

    for batch in tqdm(val_dataloader_n_refer, desc=f"Evaluating on test_n_refer"):
        input_dict = dict_to_cuda(batch)
        with torch.no_grad():
            output_dict = model.forward(images=input_dict["images"],
                                        images_clip=input_dict["images_clip"],
                                        audio_features=input_dict["audio_feats"],
                                        image_features=input_dict["image_feats"],
                                        input_ids=input_dict["input_ids"],
                                        labels=input_dict["labels"],
                                        attention_masks=input_dict["attention_masks"],
                                        masks_list=input_dict["masks"],
                                        resize_list=input_dict["resizes"],
                                        orgsize_list=input_dict["orgsizes"],
                                        conversation_list=input_dict["convs"],
                                        refs_num=input_dict["refs_num"],
                                        fids=input_dict["fids"],
                                        vids=input_dict["vids"],
                                        contrast=args.ct_weight,
                                        ref_ids=input_dict["ref_ids"],
                                        inference=True)
        pred_masks = output_dict["pred_masks"]  # list[B]:[num_seg, T, H, W]
        gt_masks = output_dict["gt_masks"]  # list[B]:[num_seg, T, H, W]
        for i in range(len(pred_masks)):
            num_seg = pred_masks[i].shape[0]
            T = pred_masks[i].shape[1]
            null_metric = utility.metric_s_for_null(pred_masks[i])

            total_metric += null_metric * num_seg * T
            count += num_seg * T


    print(f"\n  valuate on test_n_refer, metric: {total_metric/count}")

    with open(os.path.join(args.log_root, f'{args.name}.txt'), "a") as f:
        f.write(f"\n valuate on  test_n_refer:   metric {total_metric/count} \n")