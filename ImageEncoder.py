import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor
from typing import List, Tuple, Dict

"""清理GPU显存"""
import gc

class ImageEncoder(nn.Module):
    def __init__(
            self,
            device,
            model_names: List[str],
            hidden_dim: int = 768,
            k: int = 2,
            clip_vision_model=None,
            clip_processor=None
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.model_names = model_names
        self.device = device
        self.k = k
        # 设置随机种子确保结果可复现
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        # 配置模型参数
        model_configs = {
            'dino-vitb8': {
                'output_dim': 768,
                'path': "../ImgModel/dino-vitb8"
            },
            'convnextv2-tiny-1k-224': {
                'output_dim': 768,
                'path': "../ImgModel/convnextv2-tiny-1k-224"
            },
            'swin-tiny-patch4-window7-224': {
                'output_dim': 768,
                'path': "../ImgModel/swin-tiny-patch4-window7-224"
            },
            # 'clip-vit-base-patch32': {
            #     'output_dim': 768,
            #     'model': clip_vision_model,
            #     'processor': clip_processor
            # },
            'siglip-base-patch16-256': {
                'output_dim': 768,
                'model': clip_vision_model,
                'processor': clip_processor
            }
        }

        # 初始化组件
        self.processors = {}
        self.experts = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        # 加载模型和处理器
        for expert_name in model_names:
            self.clear_gpu_memory()  # 每加载一个模型前清理显存
            config = model_configs[expert_name]

            if expert_name == 'siglip-base-patch16-256':
                model = config['model']
                processor = config['processor']
            else:
                model = AutoModel.from_pretrained(config['path'])
                if expert_name == 'convnextv2-tiny-1k-224':
                    processor = AutoImageProcessor.from_pretrained(config['path'])
                else:
                    processor = AutoFeatureExtractor.from_pretrained(config['path'])

            self.experts[expert_name] = model.to(device)
            self.processors[expert_name] = processor

            # 为每个专家添加projection层
            self.projectors[expert_name] = nn.Linear(
                config['output_dim'],
                hidden_dim,
                bias=False
            ).to(device)

        # 路由器
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)

    def clear_gpu_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, images: List[Image.Image]) -> Tuple[torch.Tensor, Dict]:
        expert_features = []

        # 1. 特征提取
        for expert_name in self.model_names:
            expert = self.experts[expert_name]

            if expert_name == 'siglip-base-patch16-256':
                with torch.no_grad():
                    inputs = self.processors[expert_name](
                        images=images,
                        return_tensors="pt"
                    ).to(self.device)  # 确保输入在正确设备上
                    outputs = expert(**inputs)
                    features = outputs.pooler_output

            else:
                inputs = self.processors[expert_name](
                    images=images,
                    return_tensors="pt"
                ).to(self.device)  # 确保输入在正确设备上

                if expert_name == 'convnextv2-tiny-1k-224':
                    outputs = expert(**inputs, output_hidden_states=True)
                    features = outputs.hidden_states[-1].mean(dim=(2, 3))
                else:
                    outputs = expert(**inputs)
                    features = outputs.last_hidden_state[:, 0]

            # 确保特征在正确的设备上
            features = features.to(self.device)

            # 通过projection层
            features = self.projectors[expert_name](features)

            # 特征归一化
            features = F.normalize(features, dim=-1)

            # 再次确保特征在正确的设备上
            features = features.to(self.device)
            expert_features.append(features)

            # 清理中间变量
            del inputs, outputs
            torch.cuda.empty_cache()

        # 2. 堆叠特征 - 确保在正确的设备上
        stacked_features = torch.stack(expert_features, dim=1).to(self.device)

        # 3. 确保temperature在正确的设备上
        temperature = self.temperature.to(self.device)

        # 4. 计算相似度得分
        similarities = torch.matmul(
            stacked_features, stacked_features.transpose(1, 2)
        ) / temperature

        # 5. 计算路由概率
        # importance_scores = similarities.mean(dim=2)
        # routing_probs = F.softmax(importance_scores, dim=-1)

        # 使用sigmoid替代softmax进行路由
        importance_scores = similarities.mean(dim=2)
        routing_probs = torch.sigmoid(importance_scores)  # 使用sigmoid替代softmax

        # 可能需要添加归一化步骤
        routing_probs = routing_probs / (routing_probs.sum(dim=-1, keepdim=True) + 1e-9)

        # 6. Top-k选择
        top_k_weights, top_k_indices = torch.topk(routing_probs, self.k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # 7. 初始化组合特征张量在正确的设备上
        combined_features = torch.zeros(
            stacked_features.shape[0],
            self.hidden_dim,
            device=self.device
        )

        # 8. 特征组合
        for i in range(stacked_features.shape[0]):
            for j in range(self.k):
                expert_idx = top_k_indices[i, j]
                weight = top_k_weights[i, j]
                combined_features[i] += stacked_features[i, expert_idx] * weight

        # 9. 确保所有辅助信息在正确的设备上
        aux_info = {
            'routing_probs': routing_probs.to(self.device),
            'selected_experts': top_k_indices.to(self.device),
            'expert_weights': top_k_weights.to(self.device),
            'similarities': similarities.to(self.device),
            'expert_features': expert_features  # 已经确保在正确设备上了
        }

        # 10. 清理中间变量
        del stacked_features, similarities, importance_scores
        torch.cuda.empty_cache()

        return combined_features, aux_info


def test_image_encoder():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_names = [
        'dino-vitb8',
        'convnextv2-tiny-1k-224',
        'swin-tiny-patch4-window7-224'
    ]

    encoder = ImageEncoder(device=device, model_names=model_names, k=2)
    encoder.eval()

    img_name1 = "../../data_moe/Images_8k/667626_18933d713e.jpg"
    image1 = Image.open(img_name1).convert('RGB')
    img_name2 = "../../data_moe/Images_8k/3637013_c675de7705.jpg"
    image2 = Image.open(img_name2).convert('RGB')
    img_name3 = "../../data_moe/Images_8k/10815824_2997e03d76.jpg"
    image3 = Image.open(img_name3).convert('RGB')
    img_name4 = "../../data_moe/Images_8k/17273391_55cfc7d3d4.jpg"
    image4 = Image.open(img_name4).convert('RGB')
    img_name5 = "../../data_moe/Images_8k/19212715_20476497a3.jpg"
    image5 = Image.open(img_name5).convert('RGB')
    img_name6 = "../../data_moe/Images_8k/23445819_3a458716c1.jpg"
    image6 = Image.open(img_name6).convert('RGB')
    img_name7 = "../../data_moe/Images_8k/27782020_4dab210360.jpg"
    image7 = Image.open(img_name7).convert('RGB')
    img_name8 = "../../data_moe/Images_8k/33108590_d685bfe51c.jpg"
    image8 = Image.open(img_name8).convert('RGB')

    # 测试图像
    images = [
        image1, image2, image3, image4, image5, image6, image7, image8
    ]

    # 前向传播
    with torch.no_grad():
        features, aux_info = encoder(images)
        print(features)
        print("==========================================")

    print("\nOutput shape:", features.shape)
    print("\nRouting probabilities:", aux_info['routing_probs'])
    print("\nSelected experts:", aux_info['selected_experts'])
    print("\nSimilarity matrix:", aux_info['similarities'][0])  # 第一个样本的相似度矩阵


if __name__ == "__main__":
    test_image_encoder()