import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


class EnhancedContrastiveLoss(nn.Module):
    def __init__(
            self,
            temperature: float = 0.07,
            use_hard_negative: bool = True,
            hard_negative_weight: float = 0.15,
            margin: float = 0.3
    ):
        """
        多模态对比学习损失函数
        Args:
            temperature: 温度参数，控制相似度分布的平滑程度
            use_hard_negative: 是否使用硬负例挖掘
            hard_negative_weight: 硬负例损失的权重
            margin: 硬负例挖掘的margin
        """
        super().__init__()
        self.temperature = temperature
        self.use_hard_negative = use_hard_negative
        self.hard_negative_weight = hard_negative_weight
        self.margin = margin

    def forward(
            self,
            image_features: torch.Tensor,
            text_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:

        # 计算相似度矩阵
        logits = torch.matmul(image_features, text_features.t()) / self.temperature

        # 创建标签
        labels = torch.arange(len(image_features), device=image_features.device)

        # 计算基础对比损失
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        base_loss = (loss_i2t + loss_t2i) / 2

        total_loss = base_loss
        loss_details = {'base_loss': base_loss.item()}

        # 硬负例挖掘添加动态权重调整
        if self.use_hard_negative:
            batch_size = len(image_features)
            with torch.no_grad():
                mask = torch.eye(batch_size, device=logits.device)

                # 计算正例相似度
                pos_sim = logits.diag()

                # 找出最难的负例
                max_negative_i2t = torch.max(
                    logits - mask * 1e9,
                    dim=1
                )[0]
                max_negative_t2i = torch.max(
                    logits.t() - mask * 1e9,
                    dim=1
                )[0]

                # 动态调整硬负例权重
                pos_mean = pos_sim.mean()
                neg_mean = (max_negative_i2t.mean() + max_negative_t2i.mean()) / 2
                dynamic_weight = torch.sigmoid(pos_mean - neg_mean) * self.hard_negative_weight

            # 计算硬负例损失
            hard_negative_loss = torch.mean(
                torch.stack([
                    torch.clamp(max_negative_i2t - logits.diag() + self.margin, min=0),
                    torch.clamp(max_negative_t2i - logits.t().diag() + self.margin, min=0)
                ])
            )

            # 使用动态权重
            total_loss = total_loss + dynamic_weight * hard_negative_loss
            loss_details['hard_negative_loss'] = hard_negative_loss.item()
            loss_details['dynamic_weight'] = dynamic_weight.item()

        # 添加监控指标
        with torch.no_grad():
            loss_details.update({
                'total_loss': total_loss.item(),
                'similarity_max': logits.max().item(),
                'similarity_min': logits.min().item(),
                'similarity_mean': logits.mean().item(),
                'diagonal_similarity_mean': logits.diag().mean().item(),
                'pos_neg_ratio': (pos_sim.mean() / (
                            max_negative_i2t.mean() + max_negative_t2i.mean()) * 2).item() if self.use_hard_negative else 0
            })

        return total_loss, loss_details

    def extra_repr(self) -> str:
        """返回额外的字符串表示"""
        return (f'temperature={self.temperature}, '
                f'use_hard_negative={self.use_hard_negative}, '
                f'hard_negative_weight={self.hard_negative_weight}, '
                f'margin={self.margin}')




#
# class SigLIPLoss(nn.Module):
#     def __init__(
#             self,
#             logit_scale: float = 1.0 / 0.07,  # SigLIP使用的缩放因子，相当于1/temperature
#             use_hard_negative: bool = True,
#             hard_negative_weight: float = 0.1,
#             margin: float = 0.3
#     ):
#         """
#         SigLIP损失函数 - 使用sigmoid交叉熵替代softmax交叉熵
#         Args:
#             logit_scale: 相似度缩放因子，SigLIP的默认值通常较大（相当于较小的temperature）
#             use_hard_negative: 是否使用硬负例挖掘
#             hard_negative_weight: 硬负例损失的权重
#             margin: 硬负例挖掘的margin
#         """
#         super().__init__()
#         self.logit_scale = logit_scale
#         self.use_hard_negative = use_hard_negative
#         self.hard_negative_weight = hard_negative_weight
#         self.margin = margin
#
#     def forward(
#             self,
#             image_features: torch.Tensor,
#             text_features: torch.Tensor,
#     ) -> Tuple[torch.Tensor, Dict]:
#
#         # 计算缩放后的相似度矩阵
#         logits = self.logit_scale * torch.matmul(image_features, text_features.t())
#
#         # 创建标签矩阵 - 对角线为正例(1)，其余为负例(0)
#         batch_size = len(image_features)
#         labels = torch.eye(batch_size, device=image_features.device)
#
#         # 计算SigLIP的sigmoid交叉熵损失
#         # SigLIP对每个样本对分别计算BCE损失，而不是使用softmax
#         loss_i2t = F.binary_cross_entropy_with_logits(logits, labels)
#         loss_t2i = F.binary_cross_entropy_with_logits(logits.t(), labels)
#         base_loss = (loss_i2t + loss_t2i) / 2
#
#         total_loss = base_loss
#         loss_details = {'base_loss': base_loss.item()}
#
#         # 硬负例挖掘添加动态权重调整
#         if self.use_hard_negative:
#             with torch.no_grad():
#                 mask = torch.eye(batch_size, device=logits.device)
#
#                 # 应用sigmoid获取相似度概率
#                 similarity = torch.sigmoid(logits)
#
#                 # 计算正例相似度
#                 pos_sim = similarity.diag()
#
#                 # 找出最难的负例 (最高相似度的非对角线元素)
#                 max_negative_i2t = torch.max(
#                     similarity - mask * 1e9,
#                     dim=1
#                 )[0]
#                 max_negative_t2i = torch.max(
#                     similarity.t() - mask * 1e9,
#                     dim=1
#                 )[0]
#
#                 # 动态调整硬负例权重
#                 pos_mean = pos_sim.mean()
#                 neg_mean = (max_negative_i2t.mean() + max_negative_t2i.mean()) / 2
#                 dynamic_weight = torch.sigmoid(pos_mean - neg_mean) * self.hard_negative_weight
#
#             # 计算硬负例损失 - 应用于sigmoid概率空间
#             hard_negative_loss = torch.mean(
#                 torch.stack([
#                     torch.clamp(max_negative_i2t - similarity.diag() + self.margin, min=0),
#                     torch.clamp(max_negative_t2i - similarity.t().diag() + self.margin, min=0)
#                 ])
#             )
#
#             # 使用动态权重
#             total_loss = total_loss + dynamic_weight * hard_negative_loss
#             loss_details['hard_negative_loss'] = hard_negative_loss.item()
#             loss_details['dynamic_weight'] = dynamic_weight.item()
#
#         # 添加监控指标
#         with torch.no_grad():
#             similarity = torch.sigmoid(logits)  # 转换为概率
#             loss_details.update({
#                 'total_loss': total_loss.item(),
#                 'similarity_max': similarity.max().item(),
#                 'similarity_min': similarity.min().item(),
#                 'similarity_mean': similarity.mean().item(),
#                 'diagonal_similarity_mean': similarity.diag().mean().item(),
#                 'pos_neg_ratio': (pos_sim.mean() / (
#                         max_negative_i2t.mean() + max_negative_t2i.mean()) * 2).item() if self.use_hard_negative else 0,
#                 'logit_scale': self.logit_scale
#             })
#
#         return total_loss, loss_details
#
#     def extra_repr(self) -> str:
#         """返回额外的字符串表示"""
#         return (f'logit_scale={self.logit_scale}, '
#                 f'use_hard_negative={self.use_hard_negative}, '
#                 f'hard_negative_weight={self.hard_negative_weight}, '
#                 f'margin={self.margin}')

# def build_loss(config: Dict) -> nn.Module:
#     """
#     根据配置构建损失函数
#     Args:
#         config: 配置字典，包含损失函数的参数
#     Returns:
#         loss_fn: 损失函数实例
#     """
#     return SigLIPLoss(
#         logit_scale=config.get('logit_scale', 1.0/0.07),
#         use_hard_negative=config.get('use_hard_negative', True),
#         hard_negative_weight=config.get('hard_negative_weight', 0.1),
#         margin=config.get('margin', 0.3)
#     )

