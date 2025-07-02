import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Tuple, Dict, Optional
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

"""清理GPU显存"""
import gc

class TextEncoder(nn.Module):
    def __init__(
            self,
            device,
            model_names: List[str],
            hidden_dim: int = 768,
            k: int = 2,
            clip_text_model=None,
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
            'bge-small-en-v1_point_5': {
                'output_dim': 384,
                'path': "../textModel/bge-small-en-v1.5"
            },
            'all-MiniLM-L6-v2': {
                'output_dim': 384,
                'path': "../textModel/all-MiniLM-L6-v2"
            },
            'e5-small': {
                'output_dim': 384,
                'path': "../textModel/e5-small"
            },
            'e5-base-v2': {
                'output_dim': 768,
                'path': "../textModel/e5-base-v2"
            },
            'gte-small': {
                'output_dim': 384,
                'path': "../textModel/gte-small"
            },
            # 'clip-vit-base-patch32': {
            #     'output_dim': 512,
            #     'model': clip_text_model,
            #     'processor': clip_processor
            # },
            'siglip-base-patch16-256': {
                'output_dim': 768,
                'model': clip_text_model,
                'processor': clip_processor
            }
        }

        # 初始化组件
        self.tokenizers = {}
        self.experts = nn.ModuleDict()
        self.projectors = nn.ModuleDict()

        # 加载模型和分词器
        for model_name in model_names:
            self.clear_gpu_memory()  # 每加载一个模型前清理显存
            config = model_configs[model_name]

            if model_name == 'siglip-base-patch16-256':
                model = config['model']
                tokenizer = config['processor']
            else:
                model = AutoModel.from_pretrained(config['path'])
                tokenizer = AutoTokenizer.from_pretrained(config['path'])

            # 冻结预训练模型参数
            for param in model.parameters():
                param.requires_grad = False

            self.experts[model_name] = model.to(device)
            self.tokenizers[model_name] = tokenizer

            # 为每个专家添加projection层
            self.projectors[model_name] = nn.Linear(
                config['output_dim'],
                hidden_dim,
                bias=False
            ).to(device)

        # 路由器 - 计算特征的相似度得分
        self.temperature = nn.Parameter(torch.ones(1) * 0.1)
    def clear_gpu_memory(self):
        gc.collect()
        torch.cuda.empty_cache()

    def forward(self, texts: List[str]) -> Tuple[torch.Tensor, Dict]:
        expert_features = []

        # 1. 特征提取
        for model_name in self.model_names:
            if model_name == 'siglip-base-patch16-256':
                with torch.no_grad():
                    # siglip文本处理
                    inputs = self.tokenizers[model_name](
                        text=texts,
                        padding="max_length",
                        truncation=True,
                        max_length=64,
                        return_tensors="pt"
                    ).to(self.device)  # 确保输入在正确设备上
                    outputs = self.experts[model_name](**inputs)
                    features = outputs.pooler_output

            elif model_name == 'bge-small-en-v1.5':
                # BGE使用CLS token
                return outputs.last_hidden_state[:, 0]

            else:
                # 其他文本模型处理
                inputs = self.tokenizers[model_name](
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                ).to(self.device)  # 确保输入在正确设备上
                outputs = self.experts[model_name](**inputs)

                if model_name == 'bge-small-en-v1.5':
                    # BGE使用CLS token
                    features = outputs.last_hidden_state[:, 0]
                else:
                    features = outputs.last_hidden_state.mean(dim=1)


            # 确保特征在正确的设备上
            features = features.to(self.device)

            # 通过projection层
            features = self.projectors[model_name](features)

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

def test_text_encoder():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 定义专家模型列表
    model_names = [
        'gte-small',
        'bge-small-en-v1_point_5',
        'e5-base-v2'
    ]

    # 初始化文本编码器
    encoder = TextEncoder(device=device, model_names=model_names, hidden_dim=768)
    encoder.eval()  # 设置为评估模式

    # 准备测试文本
    test_texts = [
        "Two dogs of different breeds looking at each other on the road .",
        "Young girl with pigtails painting outside in the grass .",
        "A little girl in pink climbs a rope bridge at the park .",
        "White dog with brown ears standing near water with head turned to one side .",
        "Smiling boy in white shirt and blue jeans in front of rock wall with man in overalls behind him .",
        "Large brown dog running away from the sprinkler in the grass ."
        "Climber climbing an ice wall"
    ]

    print("\nProcessing texts:")
    for text in test_texts:
        print(f"- {text}")

    # 进行特征提取
    with torch.no_grad():
        # 提取特征
        features, aux_info = encoder(test_texts)
        print(features)


        # 检查特征值范围
        print("\nOutput shape:", features.shape)
        print("\nRouting probabilities:", aux_info['routing_probs'])
        print("\nSelected experts:", aux_info['selected_experts'])
        print("\nSimilarity matrix:", aux_info['similarities'][0])  # 第一个样本的相似度矩阵

        return True



if __name__ == "__main__":
    test_text_encoder()