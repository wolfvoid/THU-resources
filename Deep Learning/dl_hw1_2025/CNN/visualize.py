import torch
import torch.nn as nn
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from PIL import Image
import torchvision.transforms as transforms
from captum.attr import IntegratedGradients, Saliency, GuidedGradCam,LayerGradCam
from captum.attr import visualization as viz

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 导入模型
from model import ResNet18,ResNeXt18,MultiScaleResNet18,MultiScaleResNeXt18

class Visualizer:
    def __init__(self, model_name, weights_path, data_root, input_height=58, input_width=474, num_targets=4, device=None):

        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_height = input_height
        self.input_width = input_width
        self.data_root = data_root

        # 加载数据统计信息（与训练时一致）
        self.load_data_stats()

        # 加载模型
        if model_name == 'MultiScaleResNet18':
            model = MultiScaleResNet18(height=input_height, width=input_width, num_targets=num_targets)
        elif model_name == 'ResNet18':
            model = ResNet18(height=input_height, width=input_width, num_targets=num_targets)
        elif model_name == 'MultiScaleResNeXt18':
            model = MultiScaleResNeXt18(height=input_height, width=input_width, num_targets=num_targets)
        else:
            raise ValueError(f"未知模型名称: {model_name}")
        self.model = self.load_model(model, weights_path, input_height, input_width, num_targets)

        # 图像预处理,与训练时完全一致
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[self.means], std=[self.stds]),
        ])

        print(f"MultiScaleResNet18模型加载成功")
        print(f"输入尺寸: {input_height}x{input_width}, 设备: {self.device}")
        print(f"数据统计 - Mean: {self.means:.4f}, Std: {self.stds:.4f}")

    def load_data_stats(self):
        """加载训练数据的统计信息（与训练代码一致）"""
        # 加载数据
        noisy_kappa = np.load(os.path.join(self.data_root, 'WIDE12H_bin2_2arcmin_kappa_noisy_downsampled_f3.npy'))
        label = np.load(os.path.join(self.data_root, 'label.npy'))

        Ncosmo, Nsys, H, W = noisy_kappa.shape

        # 数据分割 - 与训练代码一致
        NP_idx = np.arange(Nsys)
        train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=0.2, random_state=2025)

        noisy_kappa_train = noisy_kappa[:, train_NP_idx]
        X_train = noisy_kappa_train.reshape(-1, H, W)

        # 计算统计信息（与训练时一致）
        self.means = np.mean(X_train, dtype=np.float32)
        self.stds = np.std(X_train, dtype=np.float32)

        # 加载标签scaler信息（用于逆变换）
        y_train = label[:, train_NP_idx].reshape(-1, 5)[:, :2]
        self.label_scaler = StandardScaler()
        self.label_scaler.fit(y_train)

    def load_model(self, model, weights_path, height, width, num_targets):
        """加载MultiScaleResNet18模型和权重"""

        # 加载权重
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location=self.device)

            # 处理不同的权重文件格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                elif 'model' in checkpoint:
                    model.load_state_dict(checkpoint['model'])
                else:
                    # 尝试直接加载
                    model.load_state_dict(checkpoint)
            else:
                model.load_state_dict(checkpoint)

            print(f"权重加载成功: {weights_path}")
        else:
            raise FileNotFoundError(f"权重文件不存在: {weights_path}")

        model.eval()
        model.to(self.device)
        return model

    def load_image_from_npy(self, image_index=0, dataset_type='train'):
        """
        从.npy文件中加载第i个图像

        Args:
            image_index: 图像索引
            dataset_type: 'train' 或 'val'
        """
        # 加载数据
        noisy_kappa = np.load(os.path.join(self.data_root, 'WIDE12H_bin2_2arcmin_kappa_noisy_downsampled_f3.npy'))
        label = np.load(os.path.join(self.data_root, 'label.npy'))

        Ncosmo, Nsys, H, W = noisy_kappa.shape

        # 数据分割 - 与训练代码一致
        NP_idx = np.arange(Nsys)
        train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=0.2, random_state=2025)

        if dataset_type == 'train':
            selected_idx = train_NP_idx
            noisy_kappa_selected = noisy_kappa[:, selected_idx]
            label_selected = label[:, selected_idx]
        else:  # 'val'
            selected_idx = val_NP_idx
            noisy_kappa_selected = noisy_kappa[:, selected_idx]
            label_selected = label[:, selected_idx]

        # 重塑数据
        N_selected = label_selected.shape[0] * label_selected.shape[1]
        X_selected = noisy_kappa_selected.reshape(N_selected, H, W)
        y_selected = label_selected.reshape(N_selected, 5)[:, :2]

        # 检查索引是否有效
        if image_index >= N_selected:
            raise ValueError(f"图像索引 {image_index} 超出范围，最大索引为 {N_selected-1}")

        # 获取第i个图像
        original_image = X_selected[image_index]  # 形状: (H, W)
        true_label = y_selected[image_index]      # 真实标签

        # 预处理图像（与训练时一致）
        input_tensor = self.transform(original_image).unsqueeze(0).to(self.device)

        input_tensor = input_tensor.float()  # 转换为float32

        print(f"加载图像 #{image_index} ({dataset_type}集)")
        print(f"原始图像形状: {original_image.shape}, 张量形状: {input_tensor.shape}")
        print(f"真实标签: {true_label}")

        return input_tensor, original_image, true_label

    def get_prediction(self, input_tensor, return_original_scale=True):
        """获取模型预测结果，可选择是否逆变换到原始尺度"""
        with torch.no_grad():
            means, sigmas = self.model(input_tensor)
            means_np = means.cpu().numpy().squeeze()
            sigmas_np = sigmas.cpu().numpy().squeeze()

        if return_original_scale:
            print("using original scale for prediction output")
            # 逆变换到原始尺度（与验证时一致）
            means_original = self.label_scaler.inverse_transform(means_np.reshape(1, -1)).squeeze()
            sigmas_original = sigmas_np * self.label_scaler.var_ ** 0.5

            # 应用非负先验（与验证代码一致）
            negative_mask = means_original - sigmas_original < 0
            sigmas_original[negative_mask] = means_original[negative_mask]

            print(f"预测结果 (原始尺度) - 均值: {means_original}, 标准差: {sigmas_original}")
            return means_original, sigmas_original
        else:
            print(f"预测结果 (标准化尺度) - 均值: {means_np}, 标准差: {sigmas_np}")
            return means_np, sigmas_np

    def find_conv_layers(self):
        """查找模型中所有卷积层"""
        conv_layers = {}
        for name, layer in self.model.named_modules():
            if isinstance(layer, nn.Conv2d):
                conv_layers[name] = layer
                print(f"找到卷积层: {name} -> {layer}")
        return conv_layers

    # 获取各卷积块特征
    def get_conv_block_outputs(self, input_tensor):
        """获取各卷积块的输出特征"""
        # 存储各层输出
        features = {}

        # 注册钩子
        def hook_fn(name):
            def hook(module, input, output):
                features[name] = output.detach()
            return hook

        hooks = []

        # 为关键层注册钩子
        target_layers = {
            'conv1': self.model.conv1,
            'layer1': self.model.layer1,
            'layer2': self.model.layer2,
            'layer3': self.model.layer3,
            'layer4': self.model.layer4
        }

        for name, layer in target_layers.items():
            hook = layer.register_forward_hook(hook_fn(name))
            hooks.append(hook)

        # 前向传播
        with torch.no_grad():
            _ = self.model(input_tensor)

        # 移除钩子
        for hook in hooks:
            hook.remove()

        return features

    # 计算输入图像中每个像素对特定预测目标的贡献程度（弃用）
    def generate_saliency_for_regression(self, input_tensor, target_index=0):
        """
        为回归任务生成saliency map
        可以指定关注哪个输出目标
        """
        # 定义目标函数 - 关注特定输出
        def target_fn(inputs):
            means, sigmas = self.model(inputs)
            # 可以选择关注均值或标准差
            return means[:, target_index]  # 关注第target_index个目标的均值

        # 使用Integrated Gradients
        ig = IntegratedGradients(target_fn)
        attributions = ig.attribute(input_tensor, n_steps=50)

        return attributions, f"Output_{target_index}"

    # Gradcam
    def generate_gradcam_for_regression(self, input_tensor, target_index=0, layer_name='layer4'):
        """
        为回归任务生成GradCAM热力图

        Args:
            input_tensor: 输入张量
            target_index: 目标输出索引
            layer_name: 要使用的卷积层名称
        """
        # 查找目标层
        target_layer = None
        for name, module in self.model.named_modules():
            if name == layer_name:
                target_layer = module
                break

        if target_layer is None:
            raise ValueError(f"未找到层: {layer_name}")

        # 简单的模型包装
        class ModelWrapper(nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, x):
                means, _ = self.model(x)
                return means  # 返回所有输出

        wrapped_model = ModelWrapper(self.model)
        gradcam = LayerGradCam(wrapped_model, target_layer)

        attributions = gradcam.attribute(input_tensor, target=target_index)

        # 上采样到原始图像尺寸
        attributions = torch.nn.functional.interpolate(
            attributions,
            size=(self.input_height, self.input_width),
            mode='bilinear',
            align_corners=False
        )

        return attributions, f"Output_{target_index}"

    # 可视化：saliency maps（弃用）
    def visualize_saliency_maps(self, input_tensor, original_image, true_label, means, sigmas, save_dir='results'):
        """可视化所有输出目标的saliency maps"""
        os.makedirs(save_dir, exist_ok=True)

        # 为每个输出目标生成saliency map
        n_targets = len(means)

        fig, axes = plt.subplots(n_targets, 3, figsize=(15, 5 * n_targets))
        if n_targets == 1:
            axes = axes.reshape(1, -1)

        for target_idx in range(n_targets):
            # 生成saliency map
            attributions, title = self.generate_saliency_for_regression(input_tensor, target_idx)
            attr_np = attributions.squeeze(0).cpu().detach().numpy()

            # 单通道处理
            if len(attr_np.shape) == 3:
                attr_vis = attr_np[0]  # 取第一个通道
            else:
                attr_vis = attr_np

            # 原始图像
            axes[target_idx, 0].imshow(original_image, cmap='viridis')
            axes[target_idx, 0].set_title(f'Original Image\nTarget {target_idx}')
            axes[target_idx, 0].axis('off')

            # 热力图
            im1 = axes[target_idx, 1].imshow(attr_vis, cmap='hot')
            true_val = true_label[target_idx] if target_idx < len(true_label) else 0
            axes[target_idx, 1].set_title(f'Saliency Map\nTarget {target_idx}\nTrue: {true_val:.3f}, Pred: {means[target_idx]:.3f}')
            axes[target_idx, 1].axis('off')
            plt.colorbar(im1, ax=axes[target_idx, 1])

            # 叠加图
            axes[target_idx, 2].imshow(original_image, cmap='viridis')
            im2 = axes[target_idx, 2].imshow(attr_vis, cmap='jet', alpha=0.5)
            axes[target_idx, 2].set_title(f'Overlay\nTarget {target_idx}')
            axes[target_idx, 2].axis('off')
            plt.colorbar(im2, ax=axes[target_idx, 2])

            # 保存单个结果
            plt.figure(figsize=(10, 8))
            plt.imshow(original_image, cmap='viridis')
            plt.imshow(attr_vis, cmap='jet', alpha=0.5)
            plt.title(f'Target {target_idx}\nTrue: {true_val:.3f} ± {sigmas[target_idx]:.3f}, Pred: {means[target_idx]:.3f} ± {sigmas[target_idx]:.3f}')
            plt.axis('off')
            plt.colorbar()
            plt.savefig(f'{save_dir}/target_{target_idx}_saliency.png', dpi=300, bbox_inches='tight')
            plt.close()

        plt.tight_layout()
        plt.savefig(f'{save_dir}/all_targets_saliency.png', dpi=300, bbox_inches='tight')
        plt.show()

    # 可视化：GradCAM热力图
    def visualize_gradcam_maps(self, input_tensor, original_image, true_label, means, sigmas, save_dir='results'):
        """可视化所有输出目标的GradCAM热力图"""
        os.makedirs(save_dir, exist_ok=True)

        # 为每个输出目标生成GradCAM热力图
        n_targets = len(means)

        # 尝试不同的卷积层
        layer_candidates = ['layer4', 'layer3', 'layer2', 'layer1']
        available_layers = []

        # 检查哪些层存在
        for layer_name in layer_candidates:
            for name, module in self.model.named_modules():
                if name == layer_name:
                    available_layers.append(layer_name)
                    break

        if not available_layers:
            available_layers = ['layer4']  # 默认使用layer4

        target_layer = available_layers[0]  # 使用第一个可用的层
        print(f"使用GradCAM层: {target_layer}")

        fig, axes = plt.subplots(n_targets, 3, figsize=(15, 5 * n_targets))
        if n_targets == 1:
            axes = axes.reshape(1, -1)

        for target_idx in range(n_targets):
            # 生成GradCAM热力图
            try:
                print(f"为目标 {target_idx} 生成GradCAM...")
                attributions, title = self.generate_gradcam_for_regression(input_tensor, target_idx, target_layer)

                # 处理attributions - 确保正确的形状
                if hasattr(attributions, 'dim'):
                    if attributions.dim() == 4:  # [1, 1, H, W]
                        attr_np = attributions.squeeze(0).squeeze(0).cpu().detach().numpy()
                    elif attributions.dim() == 3:  # [1, H, W]
                        attr_np = attributions.squeeze(0).cpu().detach().numpy()
                    else:
                        attr_np = attributions.cpu().detach().numpy()
                else:
                    print("attributions不是tensor，尝试转换")
                    # 如果attributions不是tensor，尝试转换
                    if isinstance(attributions, tuple) and len(attributions) > 0:
                        attr_tensor = attributions[0]
                        attr_np = attr_tensor.squeeze(0).cpu().detach().numpy()
                    else:
                        raise ValueError("无法处理attributions格式")

                print(f"GradCAM形状: {attr_np.shape}, 原始图像形状: {original_image.shape}")

                # 应用ReLU，只显示正贡献区域
                attr_np = np.maximum(attr_np, 0)

                # 归一化
                if np.max(attr_np) - np.min(attr_np) > 1e-8:
                    attr_vis = (attr_np - np.min(attr_np)) / (np.max(attr_np) - np.min(attr_np))
                else:
                    attr_vis = attr_np

                # 确保热力图与原始图像尺寸一致
                if attr_vis.shape != original_image.shape:
                    print(f"上采样热力图从 {attr_vis.shape} 到 {original_image.shape}")
                    from scipy.ndimage import zoom
                    zoom_factors = (
                        original_image.shape[0] / attr_vis.shape[0],
                        original_image.shape[1] / attr_vis.shape[1]
                    )
                    attr_vis = zoom(attr_vis, zoom_factors, order=1)  # 双线性插值

                # 原始图像
                axes[target_idx, 0].imshow(original_image, cmap='viridis')
                axes[target_idx, 0].set_title(f'Original Image\nTarget {target_idx}')
                axes[target_idx, 0].axis('off')

                # GradCAM热力图
                im1 = axes[target_idx, 1].imshow(attr_vis, cmap='hot')
                true_val = true_label[target_idx] if target_idx < len(true_label) else 0
                axes[target_idx, 1].set_title(f'GradCAM ({target_layer})\nTarget {target_idx}\nTrue: {true_val:.3f}, Pred: {means[target_idx]:.3f}')
                axes[target_idx, 1].axis('off')
                plt.colorbar(im1, ax=axes[target_idx, 1], fraction=0.046, pad=0.04)

                # 叠加图
                axes[target_idx, 2].imshow(original_image, cmap='viridis', alpha=0.7)
                im2 = axes[target_idx, 2].imshow(attr_vis, cmap='jet', alpha=0.5)
                axes[target_idx, 2].set_title(f'Overlay\nTarget {target_idx}')
                axes[target_idx, 2].axis('off')
                plt.colorbar(im2, ax=axes[target_idx, 2], fraction=0.046, pad=0.04)

                # 保存单个结果
                plt.figure(figsize=(18, 12))
                plt.imshow(original_image, cmap='viridis', alpha=0.7)
                plt.imshow(attr_vis, cmap='jet', alpha=0.5)
                plt.title(f'GradCAM - Target {target_idx} (Layer: {target_layer})\nTrue: {true_val:.3f}, Pred: {means[target_idx]:.3f} ± {sigmas[target_idx]:.3f}')
                plt.axis('off')
                plt.colorbar()
                plt.savefig(f'{save_dir}/target_{target_idx}_gradcam_{target_layer}.png', dpi=100, bbox_inches='tight')
                plt.close()

            except Exception as e:
                print(f"生成目标 {target_idx} 的GradCAM时出错: {e}")
                # 显示错误信息
                axes[target_idx, 0].text(0.5, 0.5, f'Error: {str(e)}',
                                        ha='center', va='center', transform=axes[target_idx, 0].transAxes)
                axes[target_idx, 0].set_title(f'Error for Target {target_idx}')

                for j in range(1, 3):
                    axes[target_idx, j].axis('off')

        plt.tight_layout()
        plt.savefig(f'{save_dir}/all_targets_gradcam_{target_layer}.png', dpi=100, bbox_inches='tight')
        plt.show()


    # 可视化：关键layer的conv后特征图像
    def visualize_conv_features(self, input_tensor, need_each_channel=True, save_dir='conv_features'):
        """可视化各卷积层的特征图"""
        os.makedirs(save_dir, exist_ok=True)

        # 获取各层输出特征
        features = self.get_conv_block_outputs(input_tensor)

        for layer_name, feature_maps in features.items():
            print(f"可视化层: {layer_name}, 特征形状: {feature_maps.shape}")

            # 为每层创建单独的子文件夹
            layer_save_dir = Path(save_dir) / layer_name
            layer_save_dir.mkdir(parents=True, exist_ok=True)

            # 只显示前16个通道
            display_features = 8
            feature_maps = feature_maps.squeeze(0)  # 去除batch维度

            # 创建显示用的子图（只显示前16个）
            n_rows = (display_features + 7) // 8  # 向上取整计算行数
            n_cols = min(8, display_features)

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12,12))

            # 处理axes的维度
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                axes = axes.ravel()
            else:
                axes = axes.ravel()

            # 只显示前16个通道
            for i in range(display_features):
                feature_map = feature_maps[i].cpu().numpy()

                # 归一化
                feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

                axes[i].imshow(feature_map, cmap='viridis')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')

            # 隐藏多余的子图
            for i in range(display_features, n_rows * n_cols):
                axes[i].axis('off')

            plt.suptitle(f'Layer: {layer_name} (Showing first {display_features} channels)', fontsize=16)
            plt.tight_layout()

            # 保存显示图（前16个通道的汇总图）
            plt.savefig(f'{save_dir}/{layer_name}_summary.png', dpi=100, bbox_inches='tight')
            plt.show()

            if need_each_channel:
                # 保存所有通道的单独图片
                total_channels = feature_maps.shape[0]
                print(f"保存所有{total_channels}个通道的图片到: {layer_save_dir}")

                for channel_idx in range(total_channels):
                    # 为每个通道创建单独的图片
                    plt.figure(figsize=(6, 5))
                    feature_map = feature_maps[channel_idx].cpu().numpy()

                    # 归一化
                    feature_map = (feature_map - feature_map.min()) / (feature_map.max() - feature_map.min() + 1e-8)

                    plt.imshow(feature_map, cmap='viridis')
                    plt.colorbar(label='Activation Value')
                    plt.title(f'Layer: {layer_name} - Channel {channel_idx}')
                    plt.axis('off')

                    # 保存单个通道图片
                    channel_filename = layer_save_dir / f'channel_{channel_idx:04d}.png'
                    plt.savefig(channel_filename, dpi=300, bbox_inches='tight')
                    plt.close()  # 关闭图片，避免内存泄漏

                print(f"已完成保存{total_channels}个通道图片")

    # 可视化：多尺度特征（弃用）
    def visualize_multi_scale_features(self, input_tensor, save_dir='multi_scale_features'):
        """可视化多尺度融合特征"""
        os.makedirs(save_dir, exist_ok=True)

        # 获取各层输出
        features = self.get_conv_block_outputs(input_tensor)

        # 可视化各尺度特征
        scale_layers = ['layer1', 'layer2', 'layer3', 'layer4']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        for idx, layer_name in enumerate(scale_layers):
            if layer_name in features:
                feature_tensor = features[layer_name]
                print(f"调试 - {layer_name}: 原始形状 {feature_tensor.shape}")

                # 安全地去除batch维度
                if feature_tensor.dim() == 4:  # [1, C, H, W]
                    feature_maps = feature_tensor.squeeze(0)  # [C, H, W]
                elif feature_tensor.dim() == 3:  # [C, H, W] 已经去除了batch维度
                    feature_maps = feature_tensor
                else:
                    # 处理异常情况
                    print(f"警告: {layer_name} 的维度异常: {feature_tensor.dim()}D")
                    feature_maps = feature_tensor

                print(f"调试 - {layer_name}: 处理后形状 {feature_maps.shape}")

                # 确保是三维特征图 [C, H, W]
                if feature_maps.dim() == 3 and feature_maps.shape[1] > 1 and feature_maps.shape[2] > 1:
                    # 取第一个通道
                    feature_map = feature_maps[0].cpu().numpy()

                    # 归一化
                    if np.ptp(feature_map) > 1e-8:  # ptp = max - min
                        feature_map = (feature_map - feature_map.min()) / (feature_map.ptp() + 1e-8)

                    # 显示
                    im = axes[idx].imshow(feature_map, cmap='hot')
                    axes[idx].set_title(f'{layer_name}\n{feature_map.shape}')
                    axes[idx].axis('off')
                    plt.colorbar(im, ax=axes[idx])
                else:
                    # 显示形状信息用于调试
                    axes[idx].text(0.5, 0.5, f'Shape: {feature_maps.shape}',
                                ha='center', va='center', fontsize=10)
                    axes[idx].set_title(f'{layer_name}\n无法可视化')
                    axes[idx].axis('off')

        plt.suptitle('Comparison of multi-scale feature maps', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/multi_scale_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()


    # forward进程
    def comprehensive_analysis(self, image_index=0, dataset_type='train'):
        """综合分析：saliency maps + 卷积特征"""
        if not os.path.exists('visualize'):
            os.makedirs('visualize')
        print("="*50)
        # 从.npy文件加载图像
        input_tensor, original_image, true_label = self.load_image_from_npy(image_index, dataset_type)

        # 1. 显示模型信息
        conv_layers = self.find_conv_layers()
        print(f"\n模型卷积层数量: {len(conv_layers)}")
        print("="*50)

        # 2. 获取预测
        means, sigmas = self.get_prediction(input_tensor, return_original_scale=True)

        # 3. 可视化saliency maps
        print("\n生成Saliency Maps...")
        save_dir = f'visualize/saliency_results_image{image_index}_{dataset_type}'
        self.visualize_gradcam_maps(input_tensor, original_image, true_label, means, sigmas, save_dir)
        print("="*50)

        # 4. 可视化卷积特征
        # need_each_channel=True 保存每个通道图片，False只保存前16个通道的汇总图
        print("\n可视化卷积特征...")
        save_dir = f'visualize/conv_features_results_image{image_index}_{dataset_type}'
        self.visualize_conv_features(input_tensor, need_each_channel=True, save_dir=save_dir)
        print("="*50)

        # 5. 可视化多尺度特征
        # print("\n可视化多尺度特征...")
        # save_dir = f'visualize/multi_scale_results_image{image_index}_{dataset_type}'
        # self.visualize_multi_scale_features(input_tensor, save_dir)
        # print("="*50)

        print(f"\n分析完成！所有结果已保存到相应目录")


# 使用示例
if __name__ == "__main__":
    data_root = "/home/meibingyin/Projects/DeepLearning-2025/dl_hw1_2025/CNN/dataset"  # 数据根目录
    # 所有模型均适用，但请确保模型与权重匹配
    weights_path = "/home/meibingyin/Projects/DeepLearning-2025/dl_hw1_2025/CNN/best_model_MultiScaleResNeXt18_8.67_aug_LR.pth"
    model_name = 'MultiScaleResNeXt18'  # 'MultiScaleResNet18' 或 'ResNet18'

    # 创建可视化器（参数需要与训练时一致）
    visualizer = Visualizer(
        model_name=model_name,
        weights_path=weights_path,
        data_root=data_root,
        input_height=58,
        input_width=474,
        num_targets=4,
        device = "cuda" if torch.cuda.is_available() else 'cpu'
    )

    # 分析训练集中的第0个图像
    print("分析训练集图像 #0:")
    visualizer.comprehensive_analysis(image_index=0, dataset_type='train')
