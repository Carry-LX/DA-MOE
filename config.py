def get_config():
    return {
        # 模型参数
        'clip_model_name': "../复现/clip-vit-base-patch32",

        'img_model_names': [
            'dino-vitb8',
            'convnextv2-tiny-1k-224',
            # 'swin-tiny-patch4-window7-224'
        ],

        'text_model_names': [
            # 'bge-small-en-v1_point_5',
            # 'all-MiniLM-L6-v2',
            'e5-base-v2'
        ],

        'hidden_dim': 1024,

        # 训练参数
        'epochs_stage1': 6,
        'epochs_stage2': 6,
        'batch_size': 32,
        'num_workers': 8,

        # 优化器参数
        'lr_stage1': 1e-4,
        'lr_stage2': 5e-5,
        'T0_stage1': 5,
        'T0_stage2': 5,
        'T_mult': 2,
        'eta_min_stage1': 1e-6,
        'eta_min_stage2': 5e-7,
        'min_delta': 1e-3,
        'patience': 2,  # 早停的容忍度，连续2个epoch验证损失不下降就停止

    }