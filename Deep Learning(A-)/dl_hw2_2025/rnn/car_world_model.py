import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import os

class ManualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ManualLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 输入门权重    i
        self.W_ii = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hi = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # 遗忘门权重    f
        self.W_if = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hf = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # 细胞门权重    g
        self.W_ig = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hg = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_g = nn.Parameter(torch.Tensor(hidden_size))

        # 输出门权重    o
        self.W_io = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ho = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, hidden):
        """
        Manual LSTM forward pass
        Args:
            x: input tensor of shape [seq_len, batch_size, input_size]
            hidden: tuple of (h_0, c_0), each of shape [1, batch_size, hidden_size]

        Returns:
            output: tensor of shape [seq_len, batch_size, hidden_size]
            (h_n, c_n): tuple of final hidden states
        """
        h_0, c_0 = hidden
        h_0 = h_0.squeeze(0)  # [batch_size, hidden_size]
        c_0 = c_0.squeeze(0)  # [batch_size, hidden_size]

        seq_len, batch_size, _ = x.shape
        outputs = []
        h_t = h_0
        c_t = c_0

        for t in range(seq_len):
            x_t = x[t]  # [batch_size, input_size]

            # 输入门
            i_t = torch.sigmoid(x_t @ self.W_ii + h_t @ self.W_hi + self.b_i)
            # 遗忘门
            f_t = torch.sigmoid(x_t @ self.W_if + h_t @ self.W_hf + self.b_f)
            # 细胞门
            g_t = torch.tanh(x_t @ self.W_ig + h_t @ self.W_hg + self.b_g)
            # 输出门
            o_t = torch.sigmoid(x_t @ self.W_io + h_t @ self.W_ho + self.b_o)
            # 更新细胞状态
            c_t = f_t * c_t + i_t * g_t
            # 更新隐藏状态
            h_t = o_t * torch.tanh(c_t)

            outputs.append(h_t.unsqueeze(0))

        outputs = torch.cat(outputs, dim=0)  # [seq_len, batch_size, hidden_size]
        h_n = h_t.unsqueeze(0)  # [1, batch_size, hidden_size]
        c_n = c_t.unsqueeze(0)  # [1, batch_size, hidden_size]

        return outputs, (h_n, c_n)

class WorldModel(nn.Module):
    def __init__(self, action_size, hidden_size, output_size, use_manual_lstm=True):
        super(WorldModel, self).__init__()
        self.hidden_size = hidden_size

        # CNN编码器 - 对32x32图像进行下采样
        self.cnn_encoder = nn.Sequential(
            # 输入: [batch_size, 3, 32, 32]
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 16, 16, 16]
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 32, 8, 8]
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 输出: [batch_size, 64, 4, 4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),  # 将特征压缩到128维
            nn.ReLU()
        )

        # 融合层 - 将图像特征和动作向量结合
        self.fusion_layer = nn.Linear(128 + action_size, 128)

        # NOTE 选择内置LSTM或自定义LSTM实现
        # 选择使用内置LSTM或手动实现的LSTM
        if use_manual_lstm:
            self.lstm = ManualLSTM(input_size=128, hidden_size=hidden_size)
        else:
            self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, batch_first=False)

        # 解码器 - 从LSTM输出重建图像
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)  # 输出展平的图像: 3 * 32 * 32=3072
        )

    def forward(self, state, action, hidden=None):
        """
        TODO: Implement your model here. You should implement two models, one using built-in lstm layers and one implemented by yourself.

        Forward pass for the WorldModel.
        Args:
            state: Tensor of shape [batch_size, 3, 32, 32] (current RGB image at this time step).
            action: Tensor of shape [batch_size, action_size] (3-dimensional action vector).
            hidden: Tuple of hidden states for LSTM, (h_t, c_t), each of shape [1, batch_size, hidden_size].

        Returns:
            next_state_pred: Tensor of shape [batch_size, output_size] (flattened next state prediction, 3*32*32).
            hidden: Updated hidden state tuple (h_t, c_t) for the LSTM.
        """
        batch_size = state.size(0)
        # state [batch_size, 3, 32, 32]
        image_features = self.cnn_encoder(state)  # [batch_size, 128]
        combined = torch.cat([image_features, action], dim=1)  # [batch_size, 128 + 3]
        fused_features = F.relu(self.fusion_layer(combined))  # [batch_size, 128]
        fused_features = fused_features.unsqueeze(0)  # [1, batch_size, 128]
        lstm_out, hidden = self.lstm(fused_features, hidden)  # lstm_out: [1, batch_size, hidden_size]
        lstm_out = lstm_out.squeeze(0)  # [batch_size, hidden_size]
        next_state_pred = self.decoder(lstm_out)  # [batch_size, output_size]

        return next_state_pred, hidden

class WorldModelDataLoader:
    def __init__(self, data, batch_size, sequence_length, device):
        self.data = data
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.device = device

        # 拆分数据为 train, valid, test 集合
        split_train = int(0.8 * len(self.data))
        split_valid = int(0.1 * len(self.data))
        self.train_data = self.data[:split_train]
        self.valid_data = self.data[split_train:split_train + split_valid]
        self.test_data = self.data[split_train + split_valid:]

        self.set_train()

    def set_train(self):
        self.current_data = self.train_data
        self.index = 0
        self.sub_index = 0  # 子序列的起始索引

    def set_valid(self):
        self.current_data = self.valid_data
        self.index = 0
        self.sub_index = 0

    def set_test(self):
        self.current_data = self.test_data
        self.index = 0
        self.sub_index = 0

    def get_batch(self):
        states, actions = [], []
        # data [sequences, time_steps, (state, action)] = [120, 150, (3*32*32,3) ]
        batch_data = self.current_data[self.index: self.index + self.batch_size]
        # 从当前数据集 (self.current_data) 中选择 self.batch_size 个完整的轨迹序列
        for sequence in batch_data:
            # 对于这批选中的每个轨迹序列，它不是使用整个轨迹，而是从一个内部指针 self.sub_index 开始，提取一个固定长度为 self.sequence_length 的子序列
            state_seq = [torch.tensor(step[0]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            action_seq = [torch.tensor(step[1]) for step in sequence[self.sub_index:self.sub_index + self.sequence_length]]
            if len(state_seq) < self.sequence_length:
                # 如果当前轨迹的剩余长度不足 self.sequence_length，代码会使用 零张量 (Zero Tensors) 对子序列进行填充 (pad_len)，确保每个批次内的序列长度都相同
                pad_len = self.sequence_length - len(state_seq)
                state_seq += [torch.zeros_like(state_seq[0])] * pad_len
                action_seq += [torch.zeros_like(action_seq[0])] * pad_len

            states.append(torch.stack(state_seq))
            actions.append(torch.stack(action_seq))

        self.sub_index += self.sequence_length
        # 在同一批次内的所有序列上同时前进 self.sequence_length 步
        if self.sub_index >= len(self.current_data[self.index]):
            self.index += self.batch_size
            self.sub_index = 0
        states = torch.stack(states).to(self.device)    # [batch_size, sequence_length, C, H, W]
        actions = torch.stack(actions).to(self.device)  # [batch_size, sequence_length, action_size]

        end_flag = self.index >= len(self.current_data)
        # 主索引已经超过了总数据长度， 遍历结束

        return states, actions, end_flag

with open("car_racing_data_32x32_120.pkl", "rb") as f:
    data = pickle.load(f)
print(f"Loaded data with {len(data)} sequences.")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
action_size = 3
hidden_size = 128
output_size = 32 * 32 * 3
batch_size = 16
sequence_length = 10
num_epochs = 50
learning_rate = 3e-4

data_loader = WorldModelDataLoader(data, batch_size, sequence_length, device)
# States shape: torch.Size([16, 10, 3, 32, 32])
# Actions shape: torch.Size([16, 10, 3])
model = WorldModel(action_size=action_size, hidden_size=hidden_size, output_size=output_size, use_manual_lstm=True).to(device)  # 选择是否启用手动LSTM实现
# 在模型初始化后添加
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = f"runs/world_model_experiment_{timestamp}"
writer = SummaryWriter(log_dir)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def train(num_epochs=50):
    best_val_loss = float('inf')
    epoch_pbar = tqdm(range(num_epochs), desc="Training")

    for epoch in epoch_pbar:
        model.train()
        data_loader.set_train()
        total_train_loss = 0
        total_train_samples = 0

        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)
            # Initialize the hidden state (h_0, c_0) for the LSTM, resetting it for each new batch
            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))
            # Loop through each time step in the sequence
            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)

                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                hidden = tuple([h.detach() for h in hidden])

                total_train_loss += loss.item()
                total_train_samples += 1

            if end_flag:
                break

        avg_train_loss = total_train_loss / total_train_samples
        val_loss = evaluate()

        # print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}")
        learning_rate = optimizer.param_groups[0]['lr']

        epoch_pbar.set_postfix({
            'Epoch': epoch+1,
            'Train Loss': f'{avg_train_loss:.4f}' ,
            'Val Loss': f'{val_loss:.4f}',
            'learning_rate': learning_rate
        })

        # 在计算完 avg_train_loss 和 val_loss 后添加
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning Rate', learning_rate, epoch)


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "world_model_best.pth")
            print("Best model saved.")

    writer.close()
    return

def evaluate():
    model.eval()
    data_loader.set_valid()
    total_val_loss = 0
    total_val_samples = 0

    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)

            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))

            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)

                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)

                total_val_loss += loss.item()
                total_val_samples += 1

            if end_flag:
                break

    avg_val_loss = total_val_loss / total_val_samples
    return avg_val_loss

def test():
    model.eval()
    data_loader.set_test()
    total_test_loss = 0
    total_test_samples = 0

    with torch.no_grad():
        while True:
            states, actions, end_flag = data_loader.get_batch()
            batch_size_actual = states.size(0)

            hidden = (torch.zeros(1, batch_size_actual, hidden_size).to(device),
                      torch.zeros(1, batch_size_actual, hidden_size).to(device))

            for t in range(sequence_length - 1):
                current_state = states[:, t]
                action = actions[:, t]
                next_state = states[:, t + 1].view(batch_size_actual, -1)
                next_state_pred, hidden = model(current_state, action, hidden)
                loss = criterion(next_state_pred, next_state)

                total_test_loss += loss.item()
                total_test_samples += 1

            if end_flag:
                break

    avg_test_loss = total_test_loss / total_test_samples
    print(f"Test Loss: {avg_test_loss:.4f}")

## T3 : Rollout Strategies Evaluation

# 仅用于查看数据集中的图像
def visualize_test_set_images(data_loader, max_images_per_plot=100, cols=10):
    """
    加载并可视化测试集中的所有图片，按页展示。

    Args:
        data_loader (WorldModelDataLoader): 包含测试数据的 DataLoader 实例.
        max_images_per_plot (int): 每张图最多展示的图片数量。
        cols (int): 每张图的列数。
    """
    print("--- 开始可视化测试集图片 ---")
    data_loader.set_test()
    all_images = []

    # 1. 提取所有图片
    for sequence in data_loader.current_data:
        # 每个序列中的每个时间步包含 (state, action)
        for state_action_pair in sequence:
            state = state_action_pair[0]  # 提取 state (图片)
            # state 是 numpy 数组，形状为 [3, 32, 32]
            all_images.append(state)

    total_images = len(all_images)
    print(f"测试集中共有 {total_images} 张图片。")

    if total_images == 0:
        print("测试集为空，没有图片可展示。")
        return

    num_plots = (total_images + max_images_per_plot - 1) // max_images_per_plot

    for plot_idx in range(num_plots):
        start_index = plot_idx * max_images_per_plot
        end_index = min((plot_idx + 1) * max_images_per_plot, total_images)
        images_to_plot = all_images[start_index:end_index]

        rows = (len(images_to_plot) + cols - 1) // cols

        # 调整图形大小以适应图片数量
        fig_height = rows * 1.5
        fig_width = cols * 1.5

        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        # 确保 axes 是一个2D数组，即使只有一行或一列
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1 or cols == 1:
            axes = axes.reshape(rows, cols)

        fig.suptitle(
            f"Test Set Images - Page {plot_idx + 1}/{num_plots} ({start_index + 1}-{end_index})", fontsize=16)

        for i, img_array in enumerate(images_to_plot):
            r = i // cols
            c = i % cols

            # 转换形状 [3, 32, 32] -> [32, 32, 3] for visualization
            img = np.transpose(img_array, (1, 2, 0))
            # 确保数据类型为 float 以便imshow正确处理
            img = img.astype(np.float32)
            # 归一化到 [0, 1] 范围（假设原始数据在 [0, 1] 之间）
            img = np.clip(img, 0, 1)

            axes[r, c].imshow(img)
            axes[r, c].axis('off')

        # 隐藏多余的子图
        for i in range(len(images_to_plot), rows * cols):
            r = i // cols
            c = i % cols
            axes[r, c].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # 调整布局，为 suptitle 留出空间
        # 保存图片（可选）
        plt.savefig(f'test_show/test_images_page_{plot_idx + 1}.png', dpi=200)
        # plt.show()
        print(f"Page {plot_idx + 1} displayed. Showing next page...")

    print("--- 测试集图片可视化完成 ---")

# 绘制图像对比
def visualize_predictions(ground_truth, teacher_forcing_pred, autoregressive_pred, num_steps=10, max_cols=10):
    """
    可视化对比真实帧和两种预测策略的结果 (支持自动换行)

    Args:
        ground_truth: 真实帧列表
        teacher_forcing_pred: 教师强制预测列表
        autoregressive_pred: 自回归预测列表
        num_steps: 总共要显示的步数
        max_cols: 每一行显示的图片数量 (超过此数量则换行)
    """

    # 1. 计算需要分成多少个 Block (多少“页”)
    # 例如: num_steps=25, max_cols=10 -> num_blocks=3 (10+10+5)
    num_blocks = (num_steps + max_cols - 1) // max_cols

    # 2. 动态设置图形大小和高度比例
    # 每个 Block 包含 4 行 (标题行 + 3个图像行)
    # height_ratios 需要重复 num_blocks 次: [0.2, 1, 1, 1, 0.2, 1, 1, 1, ...]
    block_ratios = [0.2, 1, 1, 1]
    total_ratios = block_ratios * num_blocks

    # 宽度基于 max_cols，高度基于 blocks 数量
    fig_width = 2.5 * min(num_steps, max_cols)
    fig_height = 7 * num_blocks

    fig, axes = plt.subplots(4 * num_blocks, max_cols,
                             figsize=(fig_width, fig_height),
                             gridspec_kw={'height_ratios': total_ratios})

    # 确保 axes 是 2D 数组，处理只有一个 Block 或只有一列的边缘情况
    if axes.ndim == 1:
        axes = axes.reshape(-1, 1)  # 变成列向量
    if num_blocks == 1 and num_steps < max_cols:
        # 如果只有不到一行，matplotlib 可能会创建一个形状为 (4, num_steps) 的 axes，
        # 但我们需要它是 (4, max_cols) 以便统一索引，不过上面的 subplots(..., max_cols) 已经保证了宽度
        pass

    strategy_names = ['Ground Truth', 'Teacher Forcing', 'Autoregressive']

    # 3. 遍历每个 Block 进行绘制
    for b in range(num_blocks):
        # 当前 Block 的起始和结束索引
        start_idx = b * max_cols
        end_idx = min((b + 1) * max_cols, num_steps)
        current_block_len = end_idx - start_idx

        # 当前 Block 在 axes 中的起始行号
        row_base = b * 4

        # 遍历当前 Block 中的每一列
        for i in range(max_cols):
            # 全局步数索引
            global_step = start_idx + i
            # 局部列索引 (在当前 subplots 中的列)
            col_idx = i

            # --- 如果已经超出了实际数据的长度，隐藏多余的坐标轴 ---
            if i >= current_block_len:
                for r in range(4):
                    axes[row_base + r, col_idx].axis('off')
                continue

            # --- 1. 绘制步骤标题 (第0行) ---
            ax_header = axes[row_base, col_idx]
            ax_header.text(0.5, 0.5, f'Step {global_step + 1}',
                           ha='center', va='center',
                           fontsize=20, fontweight='bold')
            ax_header.axis('off')

            # --- 2. 准备图片数据 ---
            # 同样的处理逻辑
            imgs = []
            # Ground Truth
            imgs.append(ground_truth[global_step].reshape(
                3, 32, 32).permute(1, 2, 0).cpu().numpy())
            # Teacher Forcing
            imgs.append(teacher_forcing_pred[global_step].reshape(
                3, 32, 32).permute(1, 2, 0).cpu().numpy())
            # Autoregressive
            imgs.append(autoregressive_pred[global_step].reshape(
                3, 32, 32).permute(1, 2, 0).cpu().numpy())

            # --- 3. 绘制三行图片 ---
            for row_offset in range(3):  # 0, 1, 2 对应 GT, TF, AR
                ax_img = axes[row_base + 1 + row_offset, col_idx]

                img_data = np.clip(imgs[row_offset], 0, 1)
                ax_img.imshow(img_data)
                ax_img.axis('off')

                # 只在当前 Block 的第一列添加策略名称
                if col_idx == 0:
                    ax_img.text(-0.2, 0.5, strategy_names[row_offset],
                                ha='right', va='center',
                                fontsize=20, fontweight='bold',
                                transform=ax_img.transAxes)

    # 调整布局
    plt.subplots_adjust(wspace=0.05, hspace=0.05,
                        left=0.15, right=0.98,  # 左边距留大一点给文字
                        top=0.95, bottom=0.05)

    plt.tight_layout()  # 可能会覆盖上面的 adjust，视情况而定

    # 保存时文件名带上步数信息
    save_name = f'rollout_comparison_steps_{num_steps}.png'
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    print(f"图像已保存为: {save_name}")
    plt.show()

def evaluate_rollout_strategies(model, test_sequence, num_steps=10):
    """
    评估两种rollout策略
    """
    model.eval()

    # 提取测试序列的前num_steps+1个时间步（需要num_steps个预测）
    states = [step[0].clone().detach().to(device) for step in test_sequence]
    actions = [step[1].clone().detach().to(device) for step in test_sequence[:num_steps]]

    teacher_forcing_preds = []
    autoregressive_preds = []

    with torch.no_grad():
        # 策略1: Teacher Forcing
        hidden_tf = (torch.zeros(1, 1, hidden_size).to(device),
                 torch.zeros(1, 1, hidden_size).to(device))
        for t in range(num_steps):
            current_state = states[t].unsqueeze(0)  # [1, 3, 32, 32]
            action = actions[t].unsqueeze(0)        # [1, 3]

            next_state_pred, hidden_tf = model(current_state, action, hidden_tf)
            teacher_forcing_preds.append(next_state_pred.squeeze(0))



        # 策略2: Autoregressive Rollout
        hidden_ar = (torch.zeros(1, 1, hidden_size).to(device),
                 torch.zeros(1, 1, hidden_size).to(device))
        # 第一步使用真实状态
        current_state = states[0].unsqueeze(0)
        for t in range(num_steps):
            action = actions[t].unsqueeze(0)
            next_state_pred, hidden_ar = model(current_state, action, hidden_ar)
            autoregressive_preds.append(next_state_pred.squeeze(0))
            # 关键区别：用预测的状态作为下一步的输入
            current_state = next_state_pred.view(1, 3, 32, 32)

    # 准备可视化数据
    ground_truth = states[1:num_steps+1]  # 真实的下一帧

    return ground_truth, teacher_forcing_preds, autoregressive_preds


def test_rollout_strategies(batch_index=0, start_index=0, prediction_length=50):
    """
    测试两种 rollout 策略 (Teacher Forcing vs Autoregressive)
    直接从原始 test_data 中读取完整序列，不再受限于 DataLoader 的 sequence_length 设置。

    Args:
        batch_index (int): 在测试集中选择第几条完整的轨迹 (0 ~ len(test_data)-1)
        start_index (int): 从轨迹的第几个时间步开始 (0 ~ 150左右)
        prediction_length (int): 需要往后预测多少步
    """
    # 1. 加载模型
    if os.path.exists("world_model_best.pth"):
        model.load_state_dict(torch.load(
            "world_model_best.pth", weights_only=True, map_location=device))
        print("已加载最佳模型权重。")

    model.eval()

    # 2. 直接访问原始测试数据列表，而不是使用 get_batch()
    # test_data 是一个 list，里面有 12 条轨迹 (120条 * 0.1)
    # 每条轨迹是一个 list，包含约 150 个 (state, action) 元组
    raw_test_data = data_loader.test_data

    # --- 边界检查 ---
    total_test_sequences = len(raw_test_data)
    if batch_index >= total_test_sequences:
        print(
            f"Error: batch_index ({batch_index}) 超出范围，测试集只有 {total_test_sequences} 条轨迹。")
        return

    # 获取选定的那条完整轨迹
    full_sequence = raw_test_data[batch_index]
    total_seq_len = len(full_sequence)
    # shape of full_sequence: 150 (3, 32, 32) (3,) <class 'float'>

    print(f"--- 数据详情 ---")
    print(f"选中轨迹索引: {batch_index}")
    print(f"轨迹总长度: {total_seq_len} 步")

    # 检查起始位置
    # 我们需要 start_index 之后的 prediction_length 步，以及再多 1 步作为 Ground Truth 的终点
    max_possible_steps = total_seq_len - start_index - 1

    if max_possible_steps <= 0:
        print(
            f"Error: start_index ({start_index}) 太大，超出轨迹长度 ({total_seq_len})")
        return

    actual_steps = min(prediction_length, max_possible_steps)

    if actual_steps < prediction_length:
        print(
            f"Warning: 轨迹剩余长度不足，预测步数从 {prediction_length} 调整为 {actual_steps}")

    print(f"测试范围: 步 {start_index} -> 步 {start_index + actual_steps}")

    # 3. 构建测试张量序列
    # 我们需要手动将 numpy 数组转换为 tensor 并放到 device 上
    test_sequence = []
    end_index = start_index + actual_steps + 1

    for i in range(start_index, end_index):
        # raw_data 中存储的是 (state_numpy, action_numpy)
        state_np, action_np = full_sequence[i][0], full_sequence[i][1]

        # 转换 State
        state_tensor = torch.from_numpy(state_np).float().to(device)
        # 转换 Action
        action_tensor = torch.from_numpy(action_np).float().to(device)

        test_sequence.append((state_tensor, action_tensor))

    # 4. 运行评估
    # evaluate_rollout_strategies 不需要修改，它接受 tensor 列表
    ground_truth, teacher_forcing_pred, autoregressive_pred = evaluate_rollout_strategies(
        model, test_sequence, num_steps=actual_steps
    )

    # 5. 计算损失
    ground_truth_flat = torch.stack([gt.view(-1) for gt in ground_truth])
    tf_preds_stack = torch.stack(teacher_forcing_pred)
    ar_preds_stack = torch.stack(autoregressive_pred)

    tf_mse = F.mse_loss(tf_preds_stack, ground_truth_flat).item()
    ar_mse = F.mse_loss(ar_preds_stack, ground_truth_flat).item()

    print(f"\n--- 结果 (Start: {start_index}, Len: {actual_steps}) ---")
    print(f"Teacher Forcing MSE: {tf_mse:.6f}")
    print(f"Autoregressive Rollout MSE: {ar_mse:.6f}")

    # 6. 可视化
    print("生成可视化...")
    visualize_predictions(ground_truth, teacher_forcing_pred,
                          autoregressive_pred, num_steps=actual_steps)

if __name__ == "__main__":
    train(num_epochs=num_epochs)
    evaluate()
    test()
    test_rollout_strategies(batch_index=2, start_index=40, prediction_length=20)
