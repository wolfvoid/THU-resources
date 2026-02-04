import torch
import torch.nn as nn
import torch.nn.functional as F
import vit
import mae
import os
import argparse
import  torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from tqdm import tqdm
import torchvision.transforms as transforms
import medmnist
from medmnist import INFO

## TODO
import matplotlib.pyplot as plt
import numpy as np


def unpatchify(x, patch_size=4):
    """
    还原 patch 为图片
    x: (B, N, patch_size**2 * 3)
    imgs: (B, 3, H, W)
    """
    p = patch_size
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]

    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs


def show_reconstruction(model, dataset, device, save_path='./reconstruction_sample.png'):
    model.eval()

    # 取几张图做样本 (这里取 4 张)
    num_samples = 4
    # 创建一个小 DataLoader 只为了取数据
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=num_samples, shuffle=True)
    imgs, _ = next(iter(loader))
    imgs = imgs.to(device)

    # 前向传播
    with torch.no_grad():
        loss, x_pred, mask = model(imgs)

    # 1. 准备原图 (Target)
    # 需要反归一化: (img * 0.5 + 0.5) 因为预处理是 mean=0.5, std=0.5
    imgs_display = imgs * 0.5 + 0.5

    # 2. 准备 Mask 图
    # mask: [B, N], 1代表被遮住, 0代表可见
    # 我们需要把 mask 变回图片形状以便可视化
    mask_display = mask.unsqueeze(-1).repeat(1, 1, 4**2 * 3)  # [B, N, P*P*3]
    mask_display = unpatchify(mask_display, patch_size=4)  # [B, 3, H, W]
    # 在原图上应用 mask (被遮挡部分画成灰色或黑色)
    # mask=1 是被遮挡，mask=0 是可见。
    # 所以可见部分是 imgs * (1-mask)，被遮挡部分设为 0.5 (灰色)
    masked_imgs = imgs_display * \
        (1 - mask_display) + mask_display * 0.8  # 0.8 浅灰色表示遮挡

    # 3. 准备重建图 (Reconstruction)
    x_pred_imgs = unpatchify(x_pred, patch_size=4)
    x_pred_imgs = x_pred_imgs * 0.5 + 0.5  # 反归一化

    # 组合: 可见部分用原图，遮挡部分用预测的
    # reconstruction = original * (1 - mask) + prediction * mask
    reconstructed_imgs = imgs_display * \
        (1 - mask_display) + x_pred_imgs * mask_display

    # --- 绘图 ---
    imgs_display = imgs_display.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    masked_imgs = masked_imgs.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()
    reconstructed_imgs = reconstructed_imgs.cpu().permute(
        0, 2, 3, 1).clamp(0, 1).numpy()

    fig, axs = plt.subplots(num_samples, 3, figsize=(9, 3*num_samples))

    # 设置列标题
    cols = ['Original', 'Masked Input', 'Reconstruction']
    for ax, col in zip(axs[0], cols):
        ax.set_title(col)

    for i in range(num_samples):
        # 原图
        axs[i, 0].imshow(imgs_display[i])
        axs[i, 0].axis('off')

        # Mask 后的输入
        axs[i, 1].imshow(masked_imgs[i])
        axs[i, 1].axis('off')

        # 重建结果
        axs[i, 2].imshow(reconstructed_imgs[i])
        axs[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Reconstruction showcase saved to {save_path}")
    plt.close()
## TODO


def run_mae_pretrain(model, train_set, optimizer, scheduler, save_dir, batch_size=64, num_epochs=20):

    device = next(model.parameters()).device

    def train_mae(model, train_loader, optimizer):
        model.train(True)
        total_loss = 0.0

        # Note: MAE uses MSE, no labels needed.
        bar = tqdm(train_loader, desc='MAE Pre-train')
        for inputs, _ in bar: # Labels are ignored during pre-training
            inputs = inputs.to(device)
            optimizer.zero_grad()

            # Forward pass: returns loss, prediction, mask
            loss, _, _ = model(inputs)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            bar.set_postfix(mae_loss=loss.item())

        epoch_loss = total_loss / len(train_loader.dataset)
        return epoch_loss

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    loss_curve = []

    for epoch in range(num_epochs):
        print('MAE Pre-train epoch:{:d}/{:d}'.format(epoch, num_epochs))
        print('*' * 100)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=10)

        train_loss = train_mae(model, train_loader, optimizer)
        scheduler.step()

        loss_curve.append(train_loss)
        print("MAE training loss: {:.4f}".format(train_loss))

    # Save the pre-trained encoder (ViT part) only
    torch.save(model.encoder.state_dict(), os.path.join(save_dir, 'mae_encoder_weights.pth'))

    print("\nMAE Pre-training complete. Reconstruction Loss Curve (for report):", loss_curve)

    show_reconstruction(model, train_set, device,
                        save_path=os.path.join(save_dir, 'mae_visualize.png'))
    return loss_curve


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='hw3')
    parser.add_argument('--save_dir', type=str, default='./mae_pretrain_checkpoints/', help='model save path')
    args = parser.parse_args()

    # data preparation
    data_flag = 'pathmnist'
    download = False
    info = INFO[data_flag]
    task = info['task']
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # download dataset first and modify the data_path accordingly
    data_path = './'
    train_dataset = DataClass(root=data_path, split='train', transform=data_transform, size=64, download=download)
    valid_dataset = DataClass(root=data_path, split='val', transform=data_transform, size=64, download=download)
    test_dataset = DataClass(root=data_path, split='test', transform=data_transform, size=64, download=download)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    # Initialize ViT encoder (without classification head)

    # you can modify the model size here if you want
    base_encoder = vit.ViT(num_classes=0, embed_dim=64, depth=2, num_heads=8)

    # Initialize MAE model given the ViT encoder
    # The masking ratio is an important hyperparameter you can tune. Enjoy!
    mae_model = mae.MaskedAutoencoder(base_encoder, decoder_dim=64, decoder_depth=1, masking_ratio=0.75)
    mae_model = mae_model.to(device)

    mae_lr = 1e-3
    mae_epochs = 10
    batch_size = 64

    mae_optimizer = optim.AdamW(mae_model.parameters(), lr=mae_lr, betas=(0.9, 0.95), weight_decay=0.05)
    mae_scheduler = CosineAnnealingLR(mae_optimizer, T_max=mae_epochs)

    recons_loss_curve = run_mae_pretrain(mae_model, train_dataset, mae_optimizer, mae_scheduler, args.save_dir, batch_size=batch_size, num_epochs=mae_epochs)
