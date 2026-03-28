"""
Audio-Visual Cross-Modal Attention Module (Stable Version)
音视频跨模态注意力模块 - 数值稳定版本
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AudioVisualCrossAttention(nn.Module):
    """
    音视频跨模态注意力模块 - 数值稳定版本

    关键改进:
    1. 移除了可能导致数值爆炸的归一化+缩放操作
    2. 在所有关键位置添加梯度裁剪
    3. 使用更保守的初始化策略
    4. 添加残差连接的权重衰减
    """

    def __init__(
        self,
        audio_dim=128,
        visual_dim=256,
        hidden_dim=256,
        num_heads=8,
        dropout=0.0,
    ):
        super().__init__()

        self.audio_dim = audio_dim
        self.visual_dim = visual_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # 音频特征投影 - 使用更小的网络
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 视觉特征投影
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # 多头交叉注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, visual_dim),
            nn.LayerNorm(visual_dim),
        )

        # 门控机制 - 使用1x1卷积
        self.gate = nn.Sequential(
            nn.Conv2d(visual_dim * 2, visual_dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """保守的权重初始化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用很小的初始化范围
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, audio_feat, visual_feat, target_frame=5):
        """
        前向传播 - FP32 精度版本 + 调试

        注意：此模块应始终保持 FP32 精度，输入会自动转换

        Args:
            audio_feat: [B, T, 128] 音频特征 (任意精度)
            visual_feat: [B*T, 256, 64, 64] SAM 视觉特征 (任意精度)
            target_frame: 目标帧索引

        Returns:
            enhanced_visual_feat: [B*T, 256, 64, 64] 增强后的特征 (与输入相同精度)
            attention_map: [B, 64, 64] 注意力图
        """

        # ===== 保存原始数据类型 =====
        original_dtype = visual_feat.dtype

        # ===== 调试：检查输入 =====
        # print(f"\n[DEBUG] AV Attention Input:")
        # print(f"  audio_feat: dtype={audio_feat.dtype}, shape={audio_feat.shape}")
        # print(f"  audio_feat: min={audio_feat.min():.6f}, max={audio_feat.max():.6f}, mean={audio_feat.mean():.6f}")
        # print(f"  visual_feat: dtype={visual_feat.dtype}, shape={visual_feat.shape}")
        # print(f"  visual_feat: min={visual_feat.min():.6f}, max={visual_feat.max():.6f}, mean={visual_feat.mean():.6f}")

        # ===== 转换输入为 FP32（模块本身已经是 FP32）=====
        audio_feat = audio_feat.float()
        visual_feat = visual_feat.float()

        B = audio_feat.shape[0]
        T = audio_feat.shape[1]
        BT, C, H, W = visual_feat.shape

        assert BT == B * T, f"维度不匹配: {BT} != {B} * {T}"

        # ===== 1. 音频特征处理 =====
        audio_feat_target = audio_feat[:, target_frame, :]  # [B, 128]
        # print(f"[DEBUG] audio_feat_target: min={audio_feat_target.min():.6f}, max={audio_feat_target.max():.6f}")

        audio_feat_target = torch.clamp(audio_feat_target, min=-10.0, max=10.0)

        # 投影到隐藏空间
        # print(f"[DEBUG] audio_proj weights: min={self.audio_proj[0].weight.min():.6f}, max={self.audio_proj[0].weight.max():.6f}")
        audio_query = self.audio_proj(audio_feat_target)  # [B, hidden_dim]
        # print(f"[DEBUG] audio_query after proj: min={audio_query.min():.6f}, max={audio_query.max():.6f}, has_nan={torch.isnan(audio_query).any()}")

        audio_query = torch.clamp(audio_query, min=-10.0, max=10.0)
        audio_query = audio_query.unsqueeze(1)  # [B, 1, hidden_dim]

        # ===== 2. 视觉特征处理 =====
        visual_feat_reshaped = visual_feat.view(B, T, C, H, W)
        visual_feat_target = visual_feat_reshaped[:, target_frame, :, :, :]  # [B, 256, 64, 64]
        # print(f"[DEBUG] visual_feat_target: min={visual_feat_target.min():.6f}, max={visual_feat_target.max():.6f}")

        # 展平空间维度
        visual_feat_flat = visual_feat_target.flatten(2).permute(0, 2, 1)  # [B, 4096, 256]

        # 投影到隐藏空间
        # print(f"[DEBUG] visual_proj weights: min={self.visual_proj[0].weight.min():.6f}, max={self.visual_proj[0].weight.max():.6f}")
        visual_kv = self.visual_proj(visual_feat_flat)  # [B, 4096, hidden_dim]
        # print(f"[DEBUG] visual_kv after proj: min={visual_kv.min():.6f}, max={visual_kv.max():.6f}, has_nan={torch.isnan(visual_kv).any()}")

        visual_kv = torch.clamp(visual_kv, min=-10.0, max=10.0)

        # ===== 3. 交叉注意力计算 =====
        # print(f"[DEBUG] Before cross_attention:")
        # print(f"  audio_query: shape={audio_query.shape}, min={audio_query.min():.6f}, max={audio_query.max():.6f}")
        # print(f"  visual_kv: shape={visual_kv.shape}, min={visual_kv.min():.6f}, max={visual_kv.max():.6f}")

        attended_feat, attention_weights = self.cross_attention(
            query=audio_query,
            key=visual_kv,
            value=visual_kv,
            need_weights=True
        )
        # print(f"[DEBUG] After cross_attention:")
        # print(f"  attended_feat: min={attended_feat.min():.6f}, max={attended_feat.max():.6f}, has_nan={torch.isnan(attended_feat).any()}")
        # print(f"  attention_weights: min={attention_weights.min():.6f}, max={attention_weights.max():.6f}, sum={attention_weights.sum(dim=-1).mean():.6f}")

        attended_feat = torch.clamp(attended_feat, min=-10.0, max=10.0)

        # ===== 4. 广播到所有空间位置 =====
        attended_feat = attended_feat.expand(-1, H * W, -1)  # [B, 4096, hidden_dim]

        # ===== 5. 投影回视觉维度 =====
        # print(f"[DEBUG] output_proj weights: min={self.output_proj[0].weight.min():.6f}, max={self.output_proj[0].weight.max():.6f}")
        enhanced_feat = self.output_proj(attended_feat)  # [B, 4096, 256]
        # print(f"[DEBUG] enhanced_feat after output_proj: min={enhanced_feat.min():.6f}, max={enhanced_feat.max():.6f}, has_nan={torch.isnan(enhanced_feat).any()}")

        enhanced_feat = torch.clamp(enhanced_feat, min=-10.0, max=10.0)

        # 恢复空间形状
        enhanced_feat = enhanced_feat.permute(0, 2, 1).view(B, C, H, W)  # [B, 256, 64, 64]

        # ===== 6. 门控融合 =====
        gate_input = torch.cat([visual_feat_target, enhanced_feat], dim=1)  # [B, 512, 64, 64]
        # print(f"[DEBUG] gate_input: min={gate_input.min():.6f}, max={gate_input.max():.6f}")

        gate = self.gate(gate_input)  # [B, 256, 64, 64]
        # print(f"[DEBUG] gate: min={gate.min():.6f}, max={gate.max():.6f}, has_nan={torch.isnan(gate).any()}")

        # 加权融合（降低增强特征的权重）
        alpha = 0.1  # 只使用10%的增强特征
        enhanced_feat_target = (1 - alpha) * visual_feat_target + alpha * gate * enhanced_feat
        # print(f"[DEBUG] enhanced_feat_target: min={enhanced_feat_target.min():.6f}, max={enhanced_feat_target.max():.6f}, has_nan={torch.isnan(enhanced_feat_target).any()}")

        # ===== 7. 放回原位置 =====
        visual_feat_reshaped_clone = visual_feat_reshaped.clone()
        visual_feat_reshaped_clone[:, target_frame, :, :, :] = enhanced_feat_target
        output_feat = visual_feat_reshaped_clone.view(BT, C, H, W)

        # ===== 8. 生成注意力图 =====
        attention_map = attention_weights.squeeze(1).view(B, H, W)  # [B, 64, 64]

        # ===== 转换输出回原始数据类型 =====
        output_feat = output_feat.to(original_dtype)
        # print(f"[DEBUG] Final output: min={output_feat.min():.6f}, max={output_feat.max():.6f}, has_nan={torch.isnan(output_feat).any()}")
        # print(f"[DEBUG] AV Attention Forward Complete\n")

        return output_feat, attention_map