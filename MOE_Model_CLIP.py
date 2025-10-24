import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from typing import List, Tuple, Dict
from transformers import AutoModel, AutoProcessor, AutoTokenizer

from Carry.A_MOE.ImageEncoder import ImageEncoder
from Carry.A_MOE.TextEncoder import TextEncoder


class EnhancedMultiModalModel(nn.Module):
    """增强型多模态模型
    结合CLIP和多专家混合(MOE)的图文特征提取和融合模型
    """

    def __init__(self,
                 img_model_names: List[str],
                 text_model_names: List[str],
                 hidden_dim: int = 1024):
        """
        Args:
            clip_model_name: CLIP模型名称
            img_model_names: 图像专家模型名称列表（不包含CLIP）
            text_model_names: 文本专家模型名称列表（不包含CLIP）
            hidden_dim: 特征维度，默认768
        """
        super().__init__()

        # self.clip_model_name = "../复现/clip-vit-base-patch32"

        self.siglip_model_name = "../复现/siglip-base-patch16-256"

        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载预训练SigLIP模型
        siglip_model = AutoModel.from_pretrained(self.siglip_model_name).to(self.device)
        siglip_processor = AutoProcessor.from_pretrained(self.siglip_model_name)
        siglip_tokenizer = AutoTokenizer.from_pretrained(self.siglip_model_name)



        # 将CLIP添加到专家列表中
        img_model_names.append('siglip-base-patch16-256')
        text_model_names.append('siglip-base-patch16-256')

        # MOE模块 - 传入CLIP视觉和文本编码器作为专家之一
        self.image_encoder = ImageEncoder(
            device=self.device,
            model_names=img_model_names,
            hidden_dim=hidden_dim,
            clip_vision_model=siglip_model.vision_model,
            clip_processor=siglip_processor
        )

        self.text_encoder = TextEncoder(
            device=self.device,
            model_names=text_model_names,
            hidden_dim=hidden_dim,
            clip_text_model=siglip_model.text_model,
            clip_processor=siglip_tokenizer
        )

    def encode_image(self, images: List[Image.Image]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码图像特征
        Args:
            images: PIL Image列表 [batch_size]

        Returns:
            image_features: 图像特征 [batch_size, hidden_dim]
        """
        # 提取图像特征
        image_features,aux_infos = self.image_encoder(images)
        # L2归一化
        image_features = F.normalize(image_features, p=2, dim=1)

        return image_features,aux_infos

    def encode_text(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码文本特征
        Args:
            texts: 文本列表 [batch_size]

        Returns:
            text_features: 文本特征 [batch_size, hidden_dim]
            expert_weights: 专家权重 [batch_size, num_experts]
        """
        # 提取文本特征
        text_features, aux_infos = self.text_encoder(texts)
        # L2归一化
        text_features = F.normalize(text_features, p=2, dim=1)

        return text_features, aux_infos

    def forward(self, images: List[Image.Image], texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        Args:
            images: PIL Image列表 [batch_size]
            texts: 文本列表 [batch_size]

        Returns:
            image_features: 图像特征 [batch_size, hidden_dim]
            text_features: 文本特征 [batch_size, hidden_dim]
            aux_outputs: 包含辅助输出的字典
        """
        # 编码图像和文本
        image_features, image_info = self.encode_image(images)
        text_features, text_info = self.encode_text(texts)

        # 返回特征和辅助输出
        aux_outputs = {
            'image_expert_infos': image_info,
            'text_expert_infos': text_info
        }

        return image_features, text_features, aux_outputs

    def get_similarity_matrix(self, image_features: torch.Tensor, text_features: torch.Tensor,
                              temperature: float = 0.1) -> torch.Tensor:
        """计算图文相似度矩阵"""
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        similarity = torch.matmul(image_features, text_features.t()) / temperature
        return similarity

    def get_similarity_matrix_recall(self, text_features: torch.Tensor, image_features: torch.Tensor,
                                     temperature: float = 0.1) -> torch.Tensor:
        """计算文本特征和图像特征之间的相似度矩阵"""
        text_features = F.normalize(text_features, p=2, dim=-1)
        image_features = F.normalize(image_features, p=2, dim=-1)
        similarity = torch.matmul(text_features, image_features.t()) / temperature
        return similarity


def test_multimodal_model():
    """测试多模态模型"""
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义模型名称
    img_model_names = [
        'dino-vitb8',
        'convnextv2-tiny-1k-224',
        # 'swin-tiny-patch4-window7-224'
    ]
    text_model_names = [
        # 'bge-small-en-v1_point_5',
        'e5-base-v2',
        'all-MiniLM-L6-v2',
        # 'e5-small'
    ]

    # 初始化模型
    model = EnhancedMultiModalModel(img_model_names, text_model_names)
    model.eval()

    # 准备测试数据
    test_images = [Image.new('RGB', (224, 224), color=f'hsl({i * 30}, 50%, 50%)')
                   for i in range(2)]
    test_texts = [
        "A colorful image with various patterns",
        "An abstract composition of shapes and colors"
    ]

    # 测试前向传播
    with torch.no_grad():

        image_features, text_features, aux_outputs = model(test_images, test_texts)

        print("\nFeature extraction successful!")
        print(f"Image features shape: {image_features.shape}")
        print(f"Text features shape: {text_features.shape}")

        # 计算相似度
        similarity = model.get_similarity_matrix(image_features, text_features)
        print(f"\nSimilarity matrix shape: {similarity.shape}")

        return True


if __name__ == "__main__":
    test_multimodal_model()