import os
import torch
import torch.optim as optim
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import json
from datetime import datetime
from tqdm import tqdm
import logging


from config import get_config
from dataset import ImageTextDataset
from MOE_Model_CLIP import EnhancedMultiModalModel
from losses import EnhancedContrastiveLoss


class EarlyStopping:
    """早停策略"""

    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.should_stop


def compute_recall_at_k(similarity: torch.Tensor, k: int) -> float:
    """计算Recall@K指标"""
    _, indices = similarity.topk(k, dim=1)
    correct = torch.arange(similarity.size(0), device=similarity.device).unsqueeze(1)
    correct = correct.repeat(1, k)
    correct = torch.eq(indices, correct).any(dim=1).float()
    return correct.mean().item()


class ModelTrainer:
    def __init__(self, model, train_loader, val_loader, save_dir, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = save_dir
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化损失函数
        self.contrastive_loss = EnhancedContrastiveLoss()

        # 初始化混合精度训练
        self.scaler = GradScaler()

        # 记录最佳验证损失
        self.best_val_loss = float('inf')
        self.best_recall = 0.0

        # 添加学习率监控
        self.lr_history = []
        self.grad_history = []

        # 初始化早停
        self.early_stopping = EarlyStopping(
            patience=config.get('patience', 5),
            min_delta=config.get('min_delta', 1e-4)
        )

        # 设置日志系统
        self._setup_loggers()

    def _setup_loggers(self):
        """配置主日志和辅助日志"""
        # 1. 配置主日志（training.log）
        log_path = os.path.join(self.save_dir, 'training.log')
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler()  # 同时输出到控制台
            ]
        )

        # 2. 配置aux_outputs专用日志
        aux_log_path = os.path.join(self.save_dir, 'aux_outputs.log')
        self.aux_logger = logging.getLogger('aux_outputs')
        self.aux_logger.setLevel(logging.INFO)

        # 创建handler并设置格式
        aux_handler = logging.FileHandler(aux_log_path)
        aux_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(message)s')
        )

        # 确保没有重复的handler
        self.aux_logger.handlers = []
        self.aux_logger.addHandler(aux_handler)

    def verify_learning_rate(self, lr):
        """验证学习率是否在合理范围内"""
        if not (5e-7 <= lr <= 1e-3):  # 修改下限为1e-6
            logging.warning(f"Learning rate {lr:.2e} is outside recommended range")
            return False
        return True

    def log_metrics(self, step, lr, grad_norm=None):
        """记录训练指标"""
        self.lr_history.append((step, lr))
        if grad_norm is not None:
            self.grad_history.append((step, grad_norm))

        if len(self.lr_history) % 100 == 0:
            logging.info(f"Step {step}: Learning rate = {lr:.2e}, Gradient norm = {grad_norm:.4f}")

    def adjust_batch_size(self, loss_value: float) -> DataLoader:
        """动态调整批处理大小"""
        if loss_value > 4.0:
            self.config['batch_size'] = max(4, self.config['batch_size'] // 2)
        elif loss_value < 1.0:
            self.config['batch_size'] = min(64, self.config['batch_size'] * 2)

        return DataLoader(
            self.train_loader.dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers'],
            collate_fn=lambda batch: (
                [item[0] for item in batch],
                [item[1] for item in batch]
            ),
            pin_memory=True,
            persistent_workers=True
        )

    # def print_model_stats(self):
    #     """打印模型参数统计"""
    #
    #     def count_parameters(module):
    #         return sum(p.numel() for p in module.parameters()), \
    #             sum(p.numel() for p in module.parameters() if p.requires_grad)
    #
    #     def print_module_stats(name, total, trainable):
    #         print(f"{name:.<50} {total:>12,} {trainable:>12,} {'√' if trainable > 0 else '×'}")
    #
    #     print("\n模型参数统计")
    #     print(f"{'模块':.<50} {'总参数':>12} {'可训练参数':>12} {'状态'}")
    #     print("-" * 80)
    #
    #
    #     # 图像编码器(MOE)
    #     print("\n图像编码器(MOE):")
    #     # 专家模型
    #     print("1. 专家模型:")
    #     for name, expert in self.model.moe_image_encoder.experts.items():
    #         total, trainable = count_parameters(expert)
    #         print_module_stats(f"   - {name}", total, trainable)
    #
    #     # 适配器
    #     print("\n2. 适配层:")
    #     for name, adapter in self.model.moe_image_encoder.adapters.items():
    #         total, trainable = count_parameters(adapter)
    #         print_module_stats(f"   - Adapter_{name}", total, trainable)
    #
    #     # 其他组件
    #     print("\n3. 路由和融合组件:")
    #     components = {
    #         'Router': self.model.moe_image_encoder.router,
    #         'Cross Attention': self.model.moe_image_encoder.cross_attention,
    #         'Feature Enhancement': self.model.moe_image_encoder.feature_enhancement,
    #         'Feature Fusion': self.model.moe_image_encoder.fusion
    #     }
    #     for name, component in components.items():
    #         total, trainable = count_parameters(component)
    #         print_module_stats(f"   - {name}", total, trainable)
    #
    #     # 文本编码器(MOE)
    #     print("\n文本编码器(MOE):")
    #     # 专家模型
    #     print("1. 专家模型:")
    #     for i, expert in enumerate(self.model.moe_text_encoder.experts):
    #         name = self.model.moe_text_encoder.model_names[i].split('/')[-1]
    #         total, trainable = count_parameters(expert)
    #         print_module_stats(f"   - {name}", total, trainable)
    #
    #     # 适配器和池化层
    #     print("\n2. 适配层和池化层:")
    #     for i in range(len(self.model.moe_text_encoder.adapters)):
    #         name = self.model.moe_text_encoder.model_names[i].split('/')[-1]
    #         adapter_total, adapter_trainable = count_parameters(self.model.moe_text_encoder.adapters[i])
    #         pooler_total, pooler_trainable = count_parameters(self.model.moe_text_encoder.poolers[i])
    #         print_module_stats(f"   - Adapter_{name}", adapter_total, adapter_trainable)
    #         print_module_stats(f"   - Pooler_{name}", pooler_total, pooler_trainable)
    #
    #     # 其他组件
    #     print("\n3. 路由和融合组件:")
    #     text_components = {
    #         'Gate': self.model.moe_text_encoder.gate,
    #         'Feature Fusion': self.model.moe_text_encoder.fusion
    #     }
    #     for name, component in text_components.items():
    #         total, trainable = count_parameters(component)
    #         print_module_stats(f"   - {name}", total, trainable)
    #
    #     # 主模型融合层
    #     print("\n特征融合层:")
    #     img_fusion_total, img_fusion_trainable = count_parameters(self.model.image_fusion)
    #     text_fusion_total, text_fusion_trainable = count_parameters(self.model.text_fusion)
    #     print_module_stats("Image Fusion", img_fusion_total, img_fusion_trainable)
    #     print_module_stats("Text Fusion", text_fusion_total, text_fusion_trainable)
    #
    #     # 总计
    #     print("\n" + "-" * 80)
    #     total_params = sum(p.numel() for p in self.model.parameters())
    #     trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    #     print_module_stats("总计", total_params, trainable_params)
    def print_model_stats(self):
        """打印模型参数统计"""

        def count_parameters(module):
            total = sum(p.numel() for p in module.parameters())
            trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
            return total, trainable

        def print_module_stats(name, total, trainable):
            print(f"{name:.<50} {total:>12,} {trainable:>12,} {'√' if trainable > 0 else '×'}")

        print("\n模型参数统计")
        print(f"{'模块':.<50} {'总参数':>12} {'可训练参数':>12} {'状态'}")
        print("-" * 80)

        # 1. 图像编码器(MOE)
        print("\n图像编码器(MOE):")
        print("1. 专家模型:")
        total_image_params = 0
        trainable_image_params = 0

        for expert_name, expert in self.model.image_encoder.experts.items():
            total, trainable = count_parameters(expert)
            total_image_params += total
            trainable_image_params += trainable
            print_module_stats(f"   - {expert_name}", total, trainable)

            # 对应的projection层
            proj_total, proj_trainable = count_parameters(self.model.image_encoder.projectors[expert_name])
            print_module_stats(f"   - {expert_name}_projector", proj_total, proj_trainable)

        # 2. 图像路由组件
        print("\n2. 图像路由组件:")
        temp_total, temp_trainable = self.model.image_encoder.temperature.numel(), \
            (self.model.image_encoder.temperature.requires_grad * self.model.image_encoder.temperature.numel())
        print_module_stats("   - Temperature Parameter", temp_total, temp_trainable)

        # 3. 文本编码器(MOE)
        print("\n文本编码器(MOE):")
        print("1. 专家模型:")
        total_text_params = 0
        trainable_text_params = 0

        for expert_name, expert in self.model.text_encoder.experts.items():
            total, trainable = count_parameters(expert)
            total_text_params += total
            trainable_text_params += trainable
            print_module_stats(f"   - {expert_name}", total, trainable)

            # 对应的projection层
            proj_total, proj_trainable = count_parameters(self.model.text_encoder.projectors[expert_name])
            print_module_stats(f"   - {expert_name}_projector", proj_total, proj_trainable)

        # 4. 文本路由组件
        print("\n2. 文本路由组件:")
        temp_total, temp_trainable = self.model.text_encoder.temperature.numel(), \
            (self.model.text_encoder.temperature.requires_grad * self.model.text_encoder.temperature.numel())
        print_module_stats("   - Temperature Parameter", temp_total, temp_trainable)

        # 5. 总计
        print("\n" + "=" * 80)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\n图像编码器总参数: {total_image_params:,} (可训练: {trainable_image_params:,})")
        print(f"文本编码器总参数: {total_text_params:,} (可训练: {trainable_text_params:,})")
        print("\n" + "-" * 80)
        print_module_stats("总计", total_params, trainable_params)

        # 6. 打印可训练参数比例
        trainable_percentage = (trainable_params / total_params) * 100 if total_params > 0 else 0
        print(f"\n可训练参数比例: {trainable_percentage:.2f}%")

    def freeze_parameters(self):

        """冻结预训练模型参数"""
        # 1. 冻结图像编码器的所有参数
        for expert_name in self.model.image_encoder.experts:
            # 冻结专家模型参数
            for param in self.model.image_encoder.experts[expert_name].parameters():
                param.requires_grad = False
            # 冻结projection层参数
            for param in self.model.image_encoder.projectors[expert_name].parameters():
                param.requires_grad = True
                # 初始化projection层参数
                if hasattr(param, 'data'):
                    torch.nn.init.xavier_uniform_(param.data)

        # 2. 冻结文本编码器的所有参数
        for expert_name in self.model.text_encoder.experts:
            # 冻结专家模型参数
            for param in self.model.text_encoder.experts[expert_name].parameters():
                param.requires_grad = False
            # 冻结projection层参数
            for param in self.model.text_encoder.projectors[expert_name].parameters():
                param.requires_grad = True
                # 初始化projection层参数
                if hasattr(param, 'data'):
                    torch.nn.init.xavier_uniform_(param.data)

        # 3. 保持temperature参数可训练
        self.model.image_encoder.temperature.requires_grad = True
        self.model.text_encoder.temperature.requires_grad = True

        print("\n=== 冻结后状态 ===")
        self.print_model_stats()

    def unfreeze_parameters(self):
        """解冻部分参数"""
        # 1. 图像编码器解冻策略
        for expert_name in self.model.image_encoder.experts:
            expert = self.model.image_encoder.experts[expert_name]

            # 首先确保所有参数都是冻结的
            for param in expert.parameters():
                param.requires_grad = False

            if 'dino' in expert_name.lower():
                if hasattr(expert, 'encoder'):
                    # 增加解冻的transformer layers数量
                    layers_to_unfreeze = 4  # 增加到4层
                    if hasattr(expert.encoder, 'layer'):
                        for layer in expert.encoder.layer[-layers_to_unfreeze:]:
                            for param in layer.parameters():
                                param.requires_grad = True

                    # 解冻所有LayerNorm层
                    for module in expert.modules():
                        if isinstance(module, nn.LayerNorm):
                            for param in module.parameters():
                                param.requires_grad = True

            elif 'convnext' in expert_name.lower():
                if hasattr(expert, 'encoder') and hasattr(expert.encoder, 'stages'):
                    # 只解冻最后一个stage
                    stage = expert.encoder.stages[-1]
                    for param in stage.parameters():
                        param.requires_grad = True

                    # 只解冻最后一个stage的downsampling_layer
                    if hasattr(stage, 'downsampling_layer'):
                        for param in stage.downsampling_layer.parameters():
                            param.requires_grad = True

                    # 有选择地解冻LayerNorm：只解冻最后一个stage的LayerNorm
                    for module in stage.modules():
                        if 'layernorm' in str(type(module).__name__).lower():
                            for param in module.parameters():
                                param.requires_grad = True

                    # 保留最后的layernorm解冻
                    if hasattr(expert, 'layernorm'):
                        for param in expert.layernorm.parameters():
                            param.requires_grad = True

            # 解冻所有projection层
            for param in self.model.image_encoder.projectors[expert_name].parameters():
                param.requires_grad = True
                if hasattr(param, 'data'):
                    torch.nn.init.xavier_uniform_(param.data)

        # 2. 文本编码器解冻策略
        for expert_name in self.model.text_encoder.experts:
            expert = self.model.text_encoder.experts[expert_name]

            # 首先确保所有参数都是冻结的
            for param in expert.parameters():
                param.requires_grad = False

            if 'bge' in expert_name.lower():
                if hasattr(expert, 'encoder'):
                    # 增加解冻的transformer layers数量
                    if hasattr(expert.encoder, 'layer'):
                        # 增加到最后3层
                        for layer in expert.encoder.layer[-3:]:
                            for param in layer.parameters():
                                param.requires_grad = True

                    # 解冻所有LayerNorm层
                    for module in expert.modules():
                        if isinstance(module, nn.LayerNorm):
                            for param in module.parameters():
                                param.requires_grad = True

                    # 解冻embedding层
                    if hasattr(expert, 'embeddings'):
                        for param in expert.embeddings.parameters():
                            param.requires_grad = True

            if 'e5-base' in expert_name.lower():
                if hasattr(expert, 'encoder'):
                    # 只解冻最后2层transformer layers
                    if hasattr(expert.encoder, 'layer'):
                        for layer in expert.encoder.layer[-3:]:
                            for param in layer.parameters():
                                param.requires_grad = True

                    # 2. 额外解冻倒数第4层的attention部分
                    if len(expert.encoder.layer) >= 4:
                        if hasattr(expert.encoder.layer[-4], 'attention'):
                            for param in expert.encoder.layer[-4].attention.parameters():
                                param.requires_grad = True

                    # 只解冻最后几个LayerNorm层
                    for i, module in enumerate(expert.modules()):
                        if isinstance(module, nn.LayerNorm):
                            # 只解冻最后25%的LayerNorm层
                            if i >= len(list(expert.modules())) * 0.65:
                                for param in module.parameters():
                                    param.requires_grad = True

                    # 保持pooler层冻结
                    if hasattr(expert, 'pooler'):
                        for param in expert.pooler.parameters():
                            param.requires_grad = False

            elif 'minilm' in expert_name.lower():
                if hasattr(expert, 'encoder'):
                    # 增加解冻的transformer layers数量
                    if hasattr(expert.encoder, 'layer'):
                        # 增加到最后2层
                        for layer in expert.encoder.layer[-1:]:
                            for param in layer.parameters():
                                param.requires_grad = True

                    # 解冻所有LayerNorm层
                    for module in expert.modules():
                        if isinstance(module, nn.LayerNorm):
                            for param in module.parameters():
                                param.requires_grad = True

                    # 解冻embedding层
                    if hasattr(expert, 'embeddings'):
                        for param in expert.embeddings.parameters():
                            param.requires_grad = True

            # 解冻所有projection层
            for param in self.model.text_encoder.projectors[expert_name].parameters():
                param.requires_grad = True
                if hasattr(param, 'data'):
                    torch.nn.init.xavier_uniform_(param.data)

        # 3. 确保routing相关参数可训练
        self.model.image_encoder.temperature.requires_grad = True
        self.model.text_encoder.temperature.requires_grad = True

        print("\n=== 解冻后状态 ===")
        self.print_model_stats()

    def train_epoch(self, epoch, optimizer, scheduler, stage):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        global_step = epoch * len(self.train_loader)

        # 调整进度条宽度和更新频率
        pbar = tqdm(self.train_loader, desc=f'Stage {stage} - Epoch {epoch + 1}', ncols=100, mininterval=0.1)
        for batch_idx, (images, texts) in enumerate(pbar):
            with autocast():
                image_features, text_features, aux_outputs = self.model(images, texts)
                main_loss, detail_loss = self.contrastive_loss(image_features, text_features)
                loss = main_loss

                # 记录aux_outputs（每100个batch记录一次）
                if batch_idx % 100 == 0:
                    # 统计每个专家被选择的次数
                    def count_expert_selections(selected_experts):
                        # 将张量展平，统计每个专家出现的次数
                        flattened = selected_experts.flatten()
                        expert_counts = torch.bincount(flattened, minlength=3)  # 3个专家，索引0,1,2
                        return expert_counts

                    image_expert_counts = count_expert_selections(aux_outputs['image_expert_infos']['selected_experts'])
                    text_expert_counts = count_expert_selections(aux_outputs['text_expert_infos']['selected_experts'])

                    self.aux_logger.info(
                        f"\nStage {stage} - Epoch {epoch + 1} - Batch {batch_idx}\n"
                        f"Image Expert Info:\n"
                        f"- Routing Probs: {aux_outputs['image_expert_infos']['routing_probs'].mean(0)}\n"
                        f"- Expert Selection Counts:\n"
                        f"  Expert 0: {image_expert_counts[0]} times\n"
                        f"  Expert 1: {image_expert_counts[1]} times\n"
                        f"  Expert 2: {image_expert_counts[2]} times\n"
                        f"Text Expert Info:\n"
                        f"- Routing Probs: {aux_outputs['text_expert_infos']['routing_probs'].mean(0)}\n"
                        f"- Expert Selection Counts:\n"
                        f"  Expert 0: {text_expert_counts[0]} times\n"
                        f"  Expert 1: {text_expert_counts[1]} times\n"
                        f"  Expert 2: {text_expert_counts[2]} times\n"
                        f"Loss: {loss.item():.4f}\n"
                        f"detail_loss: {detail_loss}\n"
                        f"{'-' * 50}"
                    )

            optimizer.zero_grad()
            self.scaler.scale(loss).backward()

            # 梯度裁剪
            self.scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=5.0,
                norm_type=2.0
            )

            self.scaler.step(optimizer)
            self.scaler.update()

            # 更新学习率 - 每个batch都更新
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            # 验证和记录指标
            self.verify_learning_rate(current_lr)
            self.log_metrics(global_step + batch_idx, current_lr, grad_norm)

            total_loss += loss.item()
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{current_lr:.2e}',  # 使用科学计数法显示学习率
                'grad': f'{grad_norm:.4f}'
            })

        return total_loss / len(self.train_loader)

    def validate(self):
        """验证模型"""
        self.model.eval()
        total_loss = 0
        all_image_features = []
        all_text_features = []

        with torch.no_grad():
            for images, texts in self.val_loader:
                image_features, text_features, aux_outputs = self.model(images, texts)

                # 累积特征
                all_image_features.append(image_features)
                all_text_features.append(text_features)

                # 计算损失
                loss,_ = self.contrastive_loss(image_features, text_features)
                total_loss += loss.item()

        # 合并所有特征
        image_features_all = torch.cat(all_image_features, dim=0)
        text_features_all = torch.cat(all_text_features, dim=0)

        # 计算整个验证集的相似度矩阵
        similarity = self.model.get_similarity_matrix(image_features_all, text_features_all)

        # 计算recall@K
        r1 = compute_recall_at_k(similarity, k=1)
        r5 = compute_recall_at_k(similarity, k=5)

        return {
            'val_loss': total_loss / len(self.val_loader),
            'recall@1': r1,
            'recall@5': r5
        }

    def train(self):
        """训练流程"""
        try:
            # 第一阶段：冻结预训练模型
            self.unfreeze_parameters()

            # 优化器和学习率调度 - 第一阶段
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['lr_stage1'],
                weight_decay=0.02,
                betas=(0.9, 0.999)
            )

            # 使用 OneCycleLR
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['lr_stage1'],
                epochs=self.config['epochs_stage1'],
                steps_per_epoch=len(self.train_loader),
                pct_start=0.1,  # 10% 的 warmup
                div_factor=25,  # 初始学习率 = max_lr/25
                final_div_factor=1e4,  # 最终学习率 = max_lr/25/1e4
                anneal_strategy='cos'  # 使用余弦退火
            )

            # 第一阶段训练
            for epoch in range(self.config['epochs_stage1']):
                train_loss = self.train_epoch(epoch, optimizer, scheduler, 1)
                val_metrics = self.validate()

                val_loss = val_metrics['val_loss']
                r1 = val_metrics['recall@1']
                r5 = val_metrics['recall@5']

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model_stage1.pth')

                if r1 > self.best_recall:
                    self.best_recall = r1
                    self.save_checkpoint('best_recall_model_stage1.pth')

                logging.info(
                    f'Stage 1 - Epoch {epoch + 1}: '
                    f'Train Loss = {train_loss:.4f}, '
                    f'Val Loss = {val_loss:.4f}, '
                    f'R@1 = {r1:.4f}, '
                    f'R@5 = {r5:.4f}'
                )

                if self.early_stopping(val_loss):
                    logging.info(f"Stage 1 - Early stopping triggered at epoch {epoch + 1}")
                    break

            # import gc
            # import torch.cuda
            #
            # # 先清理GPU内存
            # torch.cuda.empty_cache()
            # gc.collect()
            #
            # # 加载模型权重
            # model_path = "checkpoints/run_20250203_095224/best_model_stage1.pth"
            # print("Loading model weights...")
            #
            # # 临时将模型移到CPU
            # self.model = self.model.cpu()
            # checkpoint = torch.load(model_path, map_location='cpu')
            #
            # # 加载全部权重
            # self.model.load_state_dict(checkpoint['model_state_dict'])
            #
            # # 清理不需要的变量
            # del checkpoint
            # torch.cuda.empty_cache()
            # gc.collect()
            #
            # # 将模型移回GPU
            # self.model = self.model.to(self.device)
            # torch.cuda.empty_cache()
            # gc.collect()
            # print("Model weights loaded and GPU memory cleared")

            # print("Model weights loaded and GPU memory cleared")
            # model_path = "checkpoints/run_20250203_095224/best_model_stage1.pth"
            # print("Loading model weights...")
            # checkpoint = torch.load(model_path)
            # self.model.load_state_dict(checkpoint['model_state_dict'])

            # 第二阶段：微调
            # self.unfreeze_parameters()
            #
            #
            # # 修改训练集的DataLoader
            # self.train_loader = DataLoader(
            #     self.train_loader.dataset,
            #     batch_size=self.config['batch_size'] // 4,
            #     shuffle=True,
            #     num_workers=self.config['num_workers'],
            #     collate_fn=lambda batch: (
            #         [item[0] for item in batch],
            #         [item[1] for item in batch]
            #     ),
            #     pin_memory=True,
            #     persistent_workers=True
            # )
            #
            # # 第二阶段优化器和调度器
            # optimizer = optim.AdamW(
            #     self.model.parameters(),
            #     lr=self.config['lr_stage2'],
            #     weight_decay=0.02,
            #     betas=(0.9, 0.999)
            # )
            #
            # # 第二阶段的 OneCycleLR
            # scheduler = optim.lr_scheduler.OneCycleLR(
            #     optimizer,
            #     max_lr=self.config['lr_stage2'],
            #     epochs=self.config['epochs_stage2'],
            #     steps_per_epoch=len(self.train_loader),
            #     pct_start=0.1,
            #     div_factor=25,
            #     final_div_factor=1e4,
            #     anneal_strategy='cos'
            # )
            #
            # # 重置早停相关变量
            # self.early_stopping = EarlyStopping(
            #     patience=self.config.get('patience', 5),
            #     min_delta=self.config.get('min_delta', 1e-4)
            # )
            # self.best_val_loss = float('inf')
            # self.best_recall = 0.0
            #
            # # 第二阶段训练
            # for epoch in range(self.config['epochs_stage2']):
            #     train_loss = self.train_epoch(epoch, optimizer, scheduler, 2)
            #     val_metrics = self.validate()
            #
            #     val_loss = val_metrics['val_loss']
            #     r1 = val_metrics['recall@1']
            #     r5 = val_metrics['recall@5']
            #
            #     if val_loss < self.best_val_loss:
            #         self.best_val_loss = val_loss
            #         self.save_checkpoint('best_model_stage2.pth')
            #
            #     if r1 > self.best_recall:
            #         self.best_recall = r1
            #         self.save_checkpoint('best_recall_model_stage2.pth')
            #
            #     logging.info(
            #         f'Stage 2 - Epoch {epoch + 1}: '
            #         f'Train Loss = {train_loss:.4f}, '
            #         f'Val Loss = {val_loss:.4f}, '
            #         f'R@1 = {r1:.4f}, '
            #         f'R@5 = {r5:.4f}'
            #     )
            #
            #     if self.early_stopping(val_loss):
            #         logging.info(f"Stage 2 - Early stopping triggered at epoch {epoch + 1}")
            #         break

        except KeyboardInterrupt:
            logging.info("Training interrupted by user")
            self.save_checkpoint('interrupted.pth')

        except Exception as e:
            logging.exception(f"Training failed with error: {str(e)}")
            self.save_checkpoint('error.pth')

    def save_checkpoint(self, filename):
        """保存检查点"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'lr_history': self.lr_history,
            'grad_history': self.grad_history,
            'best_val_loss': self.best_val_loss,
            'best_recall': self.best_recall
        }

        torch.save(checkpoint, os.path.join(self.save_dir, filename))
        logging.info(f"Checkpoint saved: {filename}")


def main():
    # 获取配置
    config = get_config()

    # 创建保存目录
    save_dir = os.path.join('checkpoints_moe_clip', f"last_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(save_dir, exist_ok=True)

    # 保存配置
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # 初始化模型
    model = EnhancedMultiModalModel(

        img_model_names=config['img_model_names'],

        text_model_names=config['text_model_names'],
        hidden_dim=config['hidden_dim']
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    # 准备数据集和加载器
    train_dataset = ImageTextDataset(
        csv_file='../../data_moe/caption_30k/train_shuttle.csv',
        img_dir="../../data_moe/Images_30k",
    )
    val_dataset = ImageTextDataset(
        csv_file='../../data_moe/caption_30k/val_shuttle.csv',
        img_dir="../../data_moe/Images_30k",
    )

    # 数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            [item[1] for item in batch]
        ),
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=lambda batch: (
            [item[0] for item in batch],
            [item[1] for item in batch]
        ),
        pin_memory=True,
        persistent_workers=True
    )

    # 初始化训练器并开始训练
    trainer = ModelTrainer(model, train_loader, val_loader, save_dir, config)
    trainer.train()


if __name__ == "__main__":
    main()