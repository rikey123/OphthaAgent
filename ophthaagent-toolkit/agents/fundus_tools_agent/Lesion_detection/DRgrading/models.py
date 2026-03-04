import timm
import torch.nn as nn

class DRDINOv2Model(nn.Module):
    def __init__(self, model_name='vit_base_patch14_dinov2.lvd142m', num_classes=5, pretrained=False):
        super().__init__()
        # 创建模型架构（不下载预训练权重，直接使用本地微调后的权重）
        # DINOv2 的 Patch Size 是 14
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,  # 默认 False，避免从 HuggingFace 下载
            num_classes=num_classes,
            drop_rate=0.1  # 针对小数据集可以适当加 dropout
        )

    def forward(self, x):
        return self.model(x)