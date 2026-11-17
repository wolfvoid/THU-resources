import argparse
import os
import random
import time

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau,OneCycleLR
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter  # TensorBoard
from tqdm import tqdm

from model import Baseline,ResNet18,ResNet50,ConvNeXt,MSAFNet,ResNeXt18,MultiScaleResNet18,MultiScaleResNeXt18
from utils import Score, CosmologyDataset

import torch.nn.functional as F
from scipy import ndimage
import cv2
from thop import profile, clever_format # 用于计算FLOPs和参数量


class Config:
    def __init__(self, data_shape, model_name):
        self.IMG_HEIGHT = data_shape[0]
        self.IMG_WIDTH = data_shape[1]
        self.NUM_TARGETS = 4

        self.BATCH_SIZE = 16
        self.EPOCHS = 15
        # self.LEARNING_RATE = 2e-4   # 2e-4,1e-3
        self.WEIGHT_DECAY = 1e-4
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        # self.DEVICE = "cpu"
        self.use_augmentation = True   # 是否使用数据增强
        self.use_OneCycleLR = True  # 是否使用OneCycleLR调度器

        if self.use_OneCycleLR==True:
            self.LEARNING_RATE = 1e-3
        else:
            self.LEARNING_RATE = 2e-3

        # 新增：TensorBoard配置
        self.MODEL_NAME = model_name
        self.TENSORBOARD_DIR = f"runs/{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"

# 数据增强
class CosmologyTransform:
    """
    专为宇宙学数据设计的保守数据增强类
    核心思想：所有增强操作均为微小扰动，确保增强后的数据不离原始数据太远
    """
    def __init__(self, is_training=True):
        self.is_training = is_training
        self.means = None
        self.stds = None

        # 保守增强参数 - 大幅减小变化范围
        self.p_geom = 0.02      # 几何变换概率
        self.p_noise = 0.02     # 噪声添加概率
        self.p_elastic = 0.02   # 弹性形变概率
        self.p_brightness = 0.02 # 亮度调整概率

    def set_stats(self, means, stds):
        """设置标准化所需的均值和标准差"""
        self.means = means
        self.stds = stds

    def elastic_deform(self, image, alpha=12, sigma=2.5):
        """
        保守的弹性形变 - 大幅减小形变幅度
        alpha: 形变幅度从50降至12（减少76%）
        sigma: 平滑系数从5降至2.5（减少50%）
        """
        random_state = np.random.RandomState(None)
        shape = image.shape

        # 生成更小的随机位移场
        dx = random_state.rand(*shape) * 2 - 1
        dy = random_state.rand(*shape) * 2 - 1

        # 应用更强的高斯滤波，确保形变更加平滑
        dx = ndimage.gaussian_filter(dx, sigma) * alpha  # 幅度大幅减小
        dy = ndimage.gaussian_filter(dy, sigma) * alpha

        # 创建坐标网格
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

        return ndimage.map_coordinates(image, indices, order=1, mode='reflect').reshape(shape).copy()

    def correlated_noise(self, image, noise_std=0.005, correlation_scale=2.0):
        """
        添加微弱的相关噪声
        noise_std: 噪声标准差从0.02降至0.008
        """
        # 生成白噪声
        white_noise = np.random.normal(0, 1, image.shape)
        # 使用高斯滤波引入相关性
        correlated_noise = ndimage.gaussian_filter(white_noise, sigma=correlation_scale)
        # 标准化
        correlated_noise = (correlated_noise - correlated_noise.mean()) / correlated_noise.std()
        return (image + correlated_noise * noise_std).copy()

    def random_brightness_contrast(self, image, brightness_range=(0.98, 1.02)):
        """
        保守的亮度调整 - 范围大幅收紧
        从(0.9, 1.1)收紧到(0.98, 1.02)，变化幅度减小80%
        """
        if np.random.random() < self.p_brightness:
            # 在极窄范围内微调
            factor = np.random.uniform(brightness_range[0], brightness_range[1])
            image = image * factor
        return image.copy()

    def random_geometric_transform(self, image):
        """保守的几何变换 - 修复负步幅问题"""
        result = image.copy()  # 先创建副本

        # 随机水平翻转
        if np.random.random() < self.p_geom:
            result = np.fliplr(result).copy()  # 翻转后立即创建副本

        # 随机垂直翻转
        if np.random.random() < self.p_geom:
            result = np.flipud(result).copy()  # 翻转后立即创建副本

        # 90度整数倍旋转
        # if np.random.random() < self.p_geom * 0.3:
        #     k = np.random.randint(0, 4)
        #     result = np.rot90(result, k).copy()  # 旋转后立即创建副本

        return result

    def augment_image(self, image):
        """应用保守的数据增强流程"""
        if not self.is_training:
            return image.copy()

        augmented = image.copy()

        # 1. 首先应用安全的几何变换
        augmented = self.random_geometric_transform(augmented)

        # 2. 轻微弹性形变（概率很低，幅度很小）
        if np.random.random() < self.p_elastic:
            augmented = self.elastic_deform(augmented, alpha=12, sigma=2.5)

        # 3. 微调亮度/对比度
        augmented = self.random_brightness_contrast(augmented, (0.98, 1.02))

        # 4. 添加微弱的相关噪声
        if np.random.random() < self.p_noise:
            augmented = self.correlated_noise(augmented, noise_std=0.008, correlation_scale=2.0)

        return augmented

    def __call__(self, image):
        """主调用方法"""
        # 转换为numpy数组（如果是tensor）
        if isinstance(image, torch.Tensor):
            image_np = image.numpy()
        else:
            image_np = image.copy()

        # 应用保守的数据增强
        if self.is_training:
            image_np = self.augment_image(image_np)

        # 转换为tensor
        image_tensor = torch.from_numpy(image_np).float()

        # 标准化
        if self.means is not None and self.stds is not None:
            image_tensor = (image_tensor - self.means) / self.stds

        # 添加channel维度
        return image_tensor.unsqueeze(0)

    def get_conservative_parameters(self):
        """获取当前保守增强参数"""
        return {
            'p_geom': self.p_geom,
            'p_noise': self.p_noise,
            'p_elastic': self.p_elastic,
            'p_brightness': self.p_brightness,
            'elastic_alpha_range': (10, 15),      # 弹性形变幅度范围
            'brightness_range': (0.98, 1.02),    # 亮度调整范围
            'noise_std_range': (0.005, 0.01)     # 噪声强度范围
        }

# KL散度损失函数
def KL_div_posterior_loss(pred_means, pred_sigmas, truths):
    residuals_sq = (pred_means - truths) ** 2
    loss_terms = residuals_sq / (pred_sigmas ** 2)
    loss_sum = torch.sum(loss_terms, dim=1)
    log_sigma_terms = torch.sum(torch.log(pred_sigmas ** 2), dim=1)
    loss = torch.mean(loss_sum + log_sigma_terms)
    return loss

def train_epoch(model, dataloader, loss_fn, optimizer, device, epoch, use_OneCycleLR, writer=None, scheduler=None):
    model.train()
    total_loss = 0
    batch_losses = []
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Training Epoch {epoch+1}")
    for batch_idx, (X, y) in enumerate(pbar):
        X, y = X.to(device), y.to(device)

        # Forward pass
        pred_means, pred_sigmas = model(X)
        loss = loss_fn(pred_means, pred_sigmas, y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if use_OneCycleLR==True:
            scheduler.step()    # 自己添加

        total_loss += loss.item()

        # 获取当前学习率
        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]  # 获取当前学习率
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # 实时更新进度条显示当前batch的loss
        batch_losses.append(loss.item())
        pbar.set_postfix({
            'batch_loss': f'{loss.item():.6f}',
            'lr': f'{current_lr:.2e}'  # 科学计数法显示学习率
        })
        if writer is not None and batch_idx % 10 == 0:
            writer.add_scalar('Train/Batch_Loss', loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('Train/Learning_Rate', current_lr, epoch * len(dataloader) + batch_idx)

    return total_loss / len(dataloader), batch_losses

def validate_epoch(model, dataloader, loss_fn, device, epoch):
    model.eval()
    total_loss = 0
    batch_losses = []
    pbar = tqdm(dataloader, total=len(dataloader), desc=f"Validating Epoch {epoch+1}")
    with torch.no_grad():
        for X, y in pbar:
            X, y = X.to(device), y.to(device)
            pred_means, pred_sigmas = model(X)
            loss = loss_fn(pred_means, pred_sigmas, y).item()
            total_loss += loss
            batch_losses.append(loss)

            pbar.set_postfix({'batch_loss': f'{loss:.6f}'})

    return total_loss / len(dataloader), batch_losses

def initialize_model(model_name, config):
    print(f"set model = {model_name}")
    if model_name == 'Baseline':
        model = Baseline(config.IMG_HEIGHT,
                           config.IMG_WIDTH,
                           config.NUM_TARGETS).to(config.DEVICE)
    elif model_name == 'ResNet18':
        model = ResNet18(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    elif model_name == 'ResNet50':
        model = ResNet50(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    elif model_name == 'ConvNeXt':
        model = ConvNeXt(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    elif model_name == 'MSAFNet':
        model = MSAFNet(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    elif model_name == 'ResNeXt18':
        model = ResNeXt18(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    elif model_name == 'MultiScaleResNet18':
        model = MultiScaleResNet18(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    elif model_name == 'MultiScaleResNeXt18':
        model = MultiScaleResNeXt18(config.IMG_HEIGHT, config.IMG_WIDTH, config.NUM_TARGETS).to(config.DEVICE)
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model

# 计算模型的FLOPs和参数量
def calculate_flops(model_name, verbose=True):
    """
    Args:
        model: 神经网络模型
        input_shape: 输入张量的形状 (channels, height, width)
    """
    config = Config(data_shape=(474, 58), model_name=model_name)
    device = config.DEVICE
    model = initialize_model(model_name, config)
    input_shape = (1, config.IMG_HEIGHT, config.IMG_WIDTH)
    try:
        # 确保模型在评估模式
        model.eval()

        # 创建随机输入
        dummy_input = torch.randn(1, *input_shape).to(device)

        # 计算FLOPs和参数量
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)

        # 格式化输出
        flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

        result = {
            'flops': flops,
            'flops_formatted': flops_formatted,
            'params': params,
            'params_formatted': params_formatted
        }

        if verbose:
            print("=" * 60)
            print("FLOPs和参数量计算结果:")
            print(f"输入形状: {input_shape}")
            print(f"FLOPs: {flops_formatted}")
            print(f"参数量: {params_formatted}")
            print(f"原始FLOPs: {flops:.0f}")
            print(f"原始参数量: {params:.0f}")
            print("=" * 60)

        return result

    except ImportError:
        print("❌ thop库未安装，请运行: pip install thop")
        return None
    except Exception as e:
        print(f"❌ 计算FLOPs时出错: {e}")
        return None

# 单次推理时间测试
def test_inference_speed(model_path, model_name, input_shape=(1, 474, 58), num_tests=100, warmup=10):
    """
    测试模型推理速度

    Args:
        model_path: 模型文件路径
        model_name: 模型名称
        input_shape: 输入形状 (channels, height, width)
        num_tests: 测试次数
        warmup: 预热次数（不计入统计）
    """
    config = Config(data_shape=(474, 58), model_name=model_name)
    model = initialize_model(model_name, config)

    # 清空GPU缓存并重置峰值统计
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 加载权重
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE, weights_only=True))
    model.eval()

    # 创建随机输入
    batch_input = torch.randn(1, *input_shape).to(config.DEVICE)
    print(f"输入形状: {batch_input.shape}")

    # 预热（不统计时间）
    print("预热中...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(batch_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()  # 等待CUDA操作完成
        torch.cuda.reset_peak_memory_stats()  # 重置峰值统计，准备正式测试

    # 正式测试
    print(f"开始测试推理速度 ({num_tests}次)...")
    start_time = time.time()

    with torch.no_grad():
        for i in range(num_tests):
            _ = model(batch_input)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    # 获取峰值显存
    if torch.cuda.is_available():
        peak_memory_gb = torch.cuda.max_memory_allocated() / 1024**3
        print(f"峰值显存占用: {peak_memory_gb:.3f} GB")

    # 计算统计结果
    total_time = end_time - start_time
    avg_time_per_sample = total_time / num_tests
    fps = num_tests / total_time

    print("\n" + "="*50)
    print("推理速度测试结果:")
    print(f"总测试次数: {num_tests}")
    print(f"总耗时: {total_time:.4f} 秒")
    print(f"平均单次推理时间: {avg_time_per_sample*1000:.2f} 毫秒")
    print(f"推理速度: {fps:.2f} FPS (帧/秒)")
    print(f"批量吞吐量: {fps:.2f} 样本/秒")
    if torch.cuda.is_available():
        print(f"峰值显存占用: {peak_memory_gb:.3f} GB")
    else:
        print("峰值显存占用: GPU不可用")
    print("="*50)

    return {
        'total_time': total_time,
        'avg_time_ms': avg_time_per_sample * 1000,
        'fps': fps,
        'peak_memory_gb': peak_memory_gb if torch.cuda.is_available() else 0
    }


def main(args):
    # 设置随机种子以确保可重复性
    fix_seed = 2025
    torch.manual_seed(fix_seed)
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    os.environ['PYTHONHASHSEED'] = str(fix_seed) # add
    if torch.cuda.is_available():
        print("GPU available. Setting seeds for reproducibility.")
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"="*50)

    # 加载数据
    noisy_kappa = np.load(os.path.join(args.data_root, 'WIDE12H_bin2_2arcmin_kappa_noisy_downsampled_f3.npy'))
    label = np.load(os.path.join(args.data_root, 'label.npy'))
    print(f"Loaded downsampled data shape: {noisy_kappa.shape}")
    print(f"Loaded label shape: {label.shape}")
    Ncosmo, Nsys, H, W = noisy_kappa.shape
    shape = (H, W)

    # 初始化配置
    config = Config(data_shape=shape, model_name=args.model)
    print(f"Using device: {config.DEVICE}, model name: {config.MODEL_NAME}")

    # 数据分割 - 保持完全相同的参数
    NP_idx = np.arange(Nsys)
    split_fraction = 0.2

    train_NP_idx, val_NP_idx = train_test_split(NP_idx, test_size=split_fraction,
                                                random_state=fix_seed)

    noisy_kappa_train = noisy_kappa[:, train_NP_idx]
    label_train = label[:, train_NP_idx]
    noisy_kappa_val = noisy_kappa[:, val_NP_idx]
    label_val = label[:, val_NP_idx]

    Ntrain = label_train.shape[0] * label_train.shape[1]
    Nval = label_val.shape[0] * label_val.shape[1]

    print(f'Shape of the split training data = {noisy_kappa_train.shape}')
    print(f'Shape of the split validation data = {noisy_kappa_val.shape}')
    print(f'Shape of the split training labels = {label_train.shape}')
    print(f'Shape of the split validation labels = {label_val.shape}')


    # 数据重塑
    X_train = noisy_kappa_train.reshape(Ntrain, H, W)
    X_val = noisy_kappa_val.reshape(Nval, H, W)
    y_train = label_train.reshape(Ntrain, 5)[:, :2]
    y_val = label_val.reshape(Nval, 5)[:, :2]

    # 数据标准化 - 保持完全一致的处理流程
    means = np.mean(X_train, dtype=np.float32)
    stds = np.std(X_train, dtype=np.float32)
    print(f"Image stats (from train set): Mean={means}, Std={stds}")

    label_scaler = StandardScaler()
    y_train_scaled = label_scaler.fit_transform(y_train)
    y_val_scaled = label_scaler.transform(y_val)
    print(f"Label stats (from train set): Mean={label_scaler.mean_}, Std={np.sqrt(label_scaler.var_)}")

    use_augmentation = True
    if config.use_augmentation==True:
        # 使用自定义的宇宙学数据增强transform
        print("Using data augmentation.")
        cosmo_train_transform = CosmologyTransform(is_training=True)
        cosmo_train_transform.set_stats(means, stds)

        cosmo_val_transform = CosmologyTransform(is_training=False)
        cosmo_val_transform.set_stats(means, stds)
        # 创建数据集和数据加载器
        train_dataset = CosmologyDataset(
            data=X_train,
            labels=y_train_scaled,
            transform=cosmo_train_transform  # 训练集使用数据增强
        )
        val_dataset = CosmologyDataset(
            data=X_val,
            labels=y_val_scaled,
            transform=cosmo_val_transform    # 验证集不使用数据增强
        )
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[means], std=[stds]),
        ])
        # 创建数据集和数据加载器
        train_dataset = CosmologyDataset(
            data=X_train,
            labels=y_train_scaled,
            transform=transform
        )
        val_dataset = CosmologyDataset(
            data=X_val,
            labels=y_val_scaled,
            transform=transform
        )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    # 创建TensorBoard的SummaryWriter
    writer = SummaryWriter(log_dir=config.TENSORBOARD_DIR)
    print(f"TensorBoard logs will be saved to: {config.TENSORBOARD_DIR}")

    # 初始化model
    model = initialize_model(args.model, config)

    # 打印模型参数量
    print(f"="*50)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {args.model} initialized with {total_params} trainable parameters.")

    # 生成可视化图
    # conda install -c conda-forge graphviz
    # pip install torchview
    if args.only_print_structure==True:
        model.eval()
        from torchview import draw_graph
        input_tensor = torch.randn(16, 1, 58, 474).to(config.DEVICE)
        model_graph = draw_graph(model, input_size=input_tensor.shape)
        model_graph.visual_graph.render(f'model_architecture_{args.model}', format='svg')
        return

    if len(train_loader) > 0:
        sample_batch, _ = next(iter(train_loader))
        writer.add_graph(model, sample_batch.to(config.DEVICE))

    # 训练设置
    model_save_path = os.path.join(os.getcwd(), f"best_model_{args.model}.pth")
    loss_fn = KL_div_posterior_loss  # 注意：这里使用KL损失而不是MSE

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=config.LEARNING_RATE,
                                    weight_decay=config.WEIGHT_DECAY)
    if config.use_OneCycleLR==True:
        print("Using OneCycleLR scheduler.")
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE * 10,  # 大胆设大一些
            epochs=config.EPOCHS,
            steps_per_epoch=len(train_loader),  # 每个epoch包含的batch数量
            pct_start=0.3,                      # 上升阶段占总训练时间的比例    0.3
            div_factor=10,                    # 初始lr=max/div_factor
            final_div_factor=2e4,               # 最终lr=max/final_div_
            anneal_strategy='cos'               # cos/linear
        )
    else:
        scheduler = ReduceLROnPlateau(optimizer,
                                  mode='min',
                                  factor=0.5,
                                  patience=5)

    # 训练循环
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    start_time = time.time()

    # 记录每个epoch的时间和显存
    total_time = 0
    total_memory = 0

    for epoch in range(config.EPOCHS):

        epoch_start_time = time.time()

        train_loss, train_batch_losses = train_epoch(model, train_loader, loss_fn, optimizer, config.DEVICE, epoch, config.use_OneCycleLR, writer, scheduler)
        val_loss, val_batch_losses = validate_epoch(model, val_loader, loss_fn, config.DEVICE, epoch)

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        total_time += epoch_time

        # 获取当前epoch的峰值显存
        if torch.cuda.is_available():
            epoch_peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
        else:
            epoch_peak_memory = 0
        total_memory += epoch_peak_memory

        print("epoch_time = %.2f seconds, peak_memory = %.3f GB" % (epoch_time, epoch_peak_memory))

        # 记录epoch级别的loss到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)

        if config.use_OneCycleLR==False:
            scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{config.EPOCHS} |"
              f"Train Loss: {train_loss:.6f} |"
              f"Val Loss: {val_loss:.6f} |"
              f"LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> New best model saved to {model_save_path}")

    avg_epoch_time = total_time / config.EPOCHS
    avg_peak_memory = total_memory / config.EPOCHS
    print(f"\nAverage epoch time: {avg_epoch_time:.2f} seconds.")
    if torch.cuda.is_available():
        print(f"Average peak GPU memory usage: {avg_peak_memory:.3f} GB.")

    # 记录最佳loss
    writer.add_hparams({
        'batch_size': config.BATCH_SIZE,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'model': args.model
    }, {
        'hparam/best_train_loss': min(train_losses),
        'hparam/best_val_loss': best_val_loss
    })

    end_time = time.time()
    print(f"\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # 关闭TensorBoard writer
    writer.close()

    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    # Validation
    print("Validating model...")
    model.eval()
    means_pred_list, sigmas_pred_list = [], []

    with torch.no_grad():
        for X, _ in tqdm(val_loader, desc="Validation inference"):
            X = X.to(config.DEVICE)
            means_pred, sigmas_pred = model(X)
            means_pred_list.append(means_pred.cpu().numpy())
            sigmas_pred_list.append(sigmas_pred.cpu().numpy())

    mean_val = np.concatenate(means_pred_list, axis=0)
    mean_val = label_scaler.inverse_transform(mean_val)

    errorbar_val = np.concatenate(sigmas_pred_list, axis=0)
    errorbar_val = errorbar_val * label_scaler.var_ ** 0.5

    # Apply non-negative prior
    negative_mask = mean_val - errorbar_val < 0
    errorbar_val[negative_mask] = mean_val[negative_mask]

    # Calculate scores
    validation_score = Score._score_phase1(y_val, mean_val, errorbar_val)
    print(f"\nValidation Score: {np.mean(validation_score):.6f}")
    print(f"Average Error Bar: {np.mean(errorbar_val, 0)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='App Sequences Forecasting')
    parser.add_argument('--model', type=str, required=False, default='MultiScaleResNeXt18', help='model of experiment')
    # choose from Baseline,ResNet18,ResNet50,ConvNeXt,ResNeXt18,MultiScaleResNet18,MultiScaleResNeXt18
    parser.add_argument('--data_root', type=str, required=False, default='/home/meibingyin/Projects/DeepLearning-2025/dl_hw1_2025/CNN/dataset', help='data root path')
    parser.add_argument('--only_print_structure', type=str, required=False, default=False, help='data root path')
    args = parser.parse_args()

    # 训练与测试模型
    # main(args)

    # 模型参数量计算
    # complexity_info = calculate_flops(args.model, verbose=True)

    # 测试推理速度
    test_inference_speed(
        model_path=f"/home/meibingyin/Projects/DeepLearning-2025/dl_hw1_2025/CNN/best_model_MultiScaleResNeXt18_8.67_aug_LR.pth",
        model_name=args.model,
        input_shape=(1, 474, 58),
        num_tests=5000,
        warmup=500
    )

# tensorboard --logdir=runs --port=6006
