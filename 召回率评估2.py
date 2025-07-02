import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm
import warnings

from Carry.A_MOE.MOE_Model_CLIP import EnhancedMultiModalModel

from Carry.A_MOE.config import get_config

warnings.filterwarnings("ignore")


class EvalDataset(Dataset):
    """评估数据集"""

    def __init__(self, csv_path: str, img_dir: str):
        """
        CLIP数据集类

        Args:
            csv_path: 包含图像名称和文本描述的CSV文件路径
            img_dir: 图像目录路径
        """
        # 读取CSV文件
        self.df = pd.read_csv(csv_path)
        self.df = self.df.iloc[:5000]
        self.img_dir = img_dir

        # 输出当前数据集的长度
        print(f"数据集长度：{len(self.df)}")

        # 构建所有文本列表
        self.all_texts = []
        self.text_to_img_id = {}  # 记录每个文本对应的图像ID
        for idx, row in self.df.iterrows():
            img_id = idx
            for i in range(1, 6):  # 5个文本描述
                text = row[f'caption_{i}']
                self.all_texts.append(text)
                self.text_to_img_id[len(self.all_texts) - 1] = img_id

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):

        img_name = self.df.iloc[idx]['image']
        img_path = os.path.join(self.img_dir, img_name)

        # 加载并处理图像
        image = Image.open(img_path).convert('RGB')

        # image = self.transform(image)

        # 获取该图像对应的5个文本描述
        texts = [self.df.iloc[idx][f'caption_{i}'] for i in range(1, 6)]

        return  image, texts, idx


def calculate_recall(similarity_matrix, query_ids, gallery_ids, k_values=[1, 5, 10]):
    """计算召回率"""
    results = {f"R@{k}": 0.0 for k in k_values}
    n_queries = len(query_ids)

    for i in range(n_queries):
        query_id = query_ids[i]
        similarities = similarity_matrix[i]
        sorted_indices = torch.argsort(similarities, descending=True)

        # 对于每个k值计算召回率
        for k in k_values:
            top_k_indices = sorted_indices[:k]
            # 获取top-k索引对应的图像ID
            top_k_img_ids = [gallery_ids[idx.item()] for idx in top_k_indices]
            # 获取当前查询的正确图像ID
            correct_img_id = query_id if isinstance(query_id, int) else query_id.item()

            if correct_img_id in top_k_img_ids:
                results[f"R@{k}"] += 1

    # 计算平均召回率
    for k in k_values:
        results[f"R@{k}"] = results[f"R@{k}"] / n_queries * 100

    return results


def collate_fn(batch):
    """
    自定义批处理函数
    Args:
        batch: 包含(image, texts, idx)元组的列表
            - image: PIL Image对象
            - texts: 包含5个描述的列表
            - idx: 索引值
    Returns:
        images: PIL Image对象列表
        texts: 包含batch_size个列表的列表，每个内部列表包含5个文本
        indices: 包含batch_size个索引的张量
    """
    images, texts, indices = zip(*batch)

    # 将图像保持为PIL Image列表
    images = list(images)

    # 保持texts为列表的列表结构
    texts = list(texts)  # 直接转换为列表，保持嵌套结构

    # 将indices转换为张量
    indices = torch.tensor(indices)

    return images, texts, indices


def show_retrieval_results(similarity_matrix, eval_dataset, all_image_ids, top_k=10):
    """展示检索结果"""
    print("\n=== Image-to-Text Retrieval Results ===")
    # 对于每个图像
    for img_idx, img_id in enumerate(all_image_ids):
        print(f"\nImage ID: {img_id}")

        # 获取正确的文本描述
        correct_texts = eval_dataset.df.iloc[img_id][
            ['caption_1', 'caption_2', 'caption_3', 'caption_4', 'caption_5']].tolist()
        print("\nGround Truth Captions:")
        for i, text in enumerate(correct_texts, 1):
            print(f"{i}. {text}")

        # 获取相似度最高的文本
        similarities = similarity_matrix.t()[img_idx]
        top_indices = torch.argsort(similarities, descending=True)[:top_k]

        print(f"\nTop {top_k} Retrieved Captions:")
        for rank, idx in enumerate(top_indices, 1):
            text = eval_dataset.all_texts[idx.item()]
            sim_score = similarities[idx].item()
            is_correct = eval_dataset.text_to_img_id[idx.item()] == img_id
            correct_mark = "✓" if is_correct else "✗"
            print(f"{rank}. [{correct_mark}] (sim: {sim_score:.3f}) {text}")

        print("-" * 100)

        # 每5个图像后询问是否继续
        if img_idx % 5 == 4:
            response = input("\nPress Enter to continue, 'q' to quit: ")
            if response.lower() == 'q':
                break

    print("\n=== Text-to-Image Retrieval Results ===")
    # 对于每个文本
    for text_idx, text in enumerate(eval_dataset.all_texts):
        correct_img_id = eval_dataset.text_to_img_id[text_idx]
        print(f"\nQuery Text: {text}")
        print(f"Correct Image ID: {correct_img_id}")

        # 获取相似度最高的图像
        similarities = similarity_matrix[text_idx]
        top_indices = torch.argsort(similarities, descending=True)[:top_k]

        print(f"\nTop {top_k} Retrieved Images:")
        for rank, idx in enumerate(top_indices, 1):
            img_id = all_image_ids[idx.item()]
            sim_score = similarities[idx].item()
            is_correct = img_id == correct_img_id
            correct_mark = "✓" if is_correct else "✗"
            print(f"{rank}. [{correct_mark}] Image ID: {img_id} (sim: {sim_score:.3f})")

        print("-" * 100)

        # 每5个文本后询问是否继续
        if text_idx % 5 == 4:
            response = input("\nPress Enter to continue, 'q' to quit: ")
            if response.lower() == 'q':
                break


def evaluate_model(model, eval_dataset, device, batch_size=32):
    """修改后的模型评估主函数"""
    model.eval()
    data_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn
    )

    # 提取所有图像特征
    print("Extracting image features...")
    all_image_features = []
    all_image_ids = []

    with torch.no_grad():
        for images, _, ids in tqdm(data_loader):
            image_features,_ = model.encode_image(images=images)
            # image_features = data1['moe_features']
            all_image_features.append(image_features.cpu())
            all_image_ids.extend(ids.tolist())

    all_image_features = torch.cat(all_image_features, dim=0)

    # 提取所有文本特征
    print("Extracting text features...")
    all_text_features = []
    text_batch_size = 64

    text_batches = [eval_dataset.all_texts[i:i + text_batch_size]
                    for i in range(0, len(eval_dataset.all_texts), text_batch_size)]

    with torch.no_grad():
        for texts in tqdm(text_batches):
            text_features,_ = model.encode_text(texts=texts)
            # text_features = data2['moe_features']
            all_text_features.append(text_features.cpu())

    all_text_features = torch.cat(all_text_features, dim=0)

    # 计算相似度矩阵
    print("Computing similarity matrix...")
    similarity_matrix = model.get_similarity_matrix_recall(all_text_features, all_image_features, temperature=0.1)

    # 文本到图像的召回率
    print("Computing text-to-image recall...")
    t2i_recall = calculate_recall(
        similarity_matrix,
        query_ids=[eval_dataset.text_to_img_id[i] for i in range(len(eval_dataset.all_texts))],
        gallery_ids=all_image_ids,
        k_values=[1, 5, 10]
    )

    # 图像到文本的召回率
    print("Computing image-to-text recall...")
    i2t_recall = calculate_recall(
        similarity_matrix.t(),
        query_ids=all_image_ids,
        gallery_ids=[eval_dataset.text_to_img_id[i] for i in range(len(eval_dataset.all_texts))],
        k_values=[1, 5, 10]
    )

    # 展示详细的检索结果
    show_retrieval_results(similarity_matrix, eval_dataset, all_image_ids)

    return {
        "image_to_text": i2t_recall,
        "text_to_image": t2i_recall
    }


def run_evaluation(model_path, csv_path, img_dir):
    """运行评估"""
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    print("Initializing model...")

    # 初始化模型
    model = EnhancedMultiModalModel(
        img_model_names=config['img_model_names'],
        text_model_names=config['text_model_names'],
        hidden_dim=config['hidden_dim']
    ).to(device)

    # 加载模型权重
    print("Loading model weights...")
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 创建评估数据集
    print("Creating evaluation dataset...")
    eval_dataset = EvalDataset(csv_path, img_dir)

    # 运行评估
    print("Starting evaluation...")
    results = evaluate_model(model, eval_dataset, device)

    # 打印结果
    print("\nEvaluation Results:")
    print("=" * 50)
    print("Image-to-Text Retrieval:")
    for k in [1, 5, 10]:
        print(f"R@{k}: {results['image_to_text'][f'R@{k}']:.2f}%")

    print("\nText-to-Image Retrieval:")
    for k in [1, 5, 10]:
        print(f"R@{k}: {results['text_to_image'][f'R@{k}']:.2f}%")


if __name__ == "__main__":
    run_evaluation(
        model_path="checkpoints_moe_clip/last_20250702_002833/best_model_stage1.pth",
        csv_path="../../data_moe/caption_30k/test_win.csv",
        img_dir="../../data_moe/Images_30k",
    )