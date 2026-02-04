import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class TrainingVisualizer:
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        # self.test_losses = []
        self.current_epoch = 0
        self.start_time = datetime.now()

    def update(self, train_loss, val_loss, epoch):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        # self.test_losses.append(test_loss)
        self.current_epoch = epoch

    def create_plots(self):
        # 创建损失曲线图
        plt.figure(figsize=(12, 5))

        # 损失曲线
        plt.subplot(1, 2, 1)
        epochs = range(1, len(self.train_losses)+1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        # plt.plot(epochs, self.test_losses, 'g-', label='Test Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # 训练信息
        plt.subplot(1, 2, 2)
        plt.axis('off')
        info_text = (
            f"Training Progress\n"
            f"Epoch: {self.current_epoch}/{len(self.train_losses)}\n"
            f"Latest Train Loss: {self.train_losses[-1]:.4f}\n"
            f"Latest Val Loss: {self.val_losses[-1]:.4f}\n"
            f"Elapsed Time: {datetime.now() - self.start_time}"
        )
        plt.text(0.1, 0.5, info_text, fontsize=12)

        plt.tight_layout()
        return plt
