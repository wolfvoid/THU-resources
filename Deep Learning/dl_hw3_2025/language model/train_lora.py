import click
import torch
import loralib as lora
from trainers import LORATrainer
from gpt import GPT
from configs import get_configs
from dataset import EYLSFTStaticDataset

def train(ckpt, batch_size, exp_name):
    device = 'cuda'
    cfg = get_configs("gpt2/dropout")

    # === LoRA Specific Configs ===
    # 输出lora参数
    print(f"use_lora= {cfg.use_lora}")

    # =============================

    cfg.num_steps = 200000 // batch_size
    cfg.batch_size = batch_size
    cfg.ckpt = ckpt
    cfg.exp_name = exp_name

    # 1. 加载预训练模型
    model = GPT.from_pretrained(cfg, ckpt)

    # 2. 核心步骤：应用 LoRA 掩码，冻结非 LoRA 参数
    lora.mark_only_lora_as_trainable(model)

    # (可选) 打印参数量对比，确认 LoRA 生效
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[LoRA Init] Total params: {total_params:,}")
    print(f"[LoRA Init] Trainable params: {trainable_params:,}")
    print(f"[LoRA Init] Trainable ratio: {trainable_params/total_params:.4%}\n")

    # 3. 准备数据集
    train_ds = EYLSFTStaticDataset(block_size=1024,
                                   split='train',
                                   max_examples=None)
    test_ds = EYLSFTStaticDataset(block_size=1024,
                                  split='test',
                                  max_examples=None)

    # 4. 使用 LORATrainer
    trainer = LORATrainer(cfg, device, model, train_ds, test_ds)
    trainer.fit()


@click.command()
@click.option('--ckpt', '-c', default="./checkpoints/gpt2_pretrained.pt")
@click.option('--batch-size', '-b', default=1)
@click.option('--exp-name', '-n', default="lora_batchsize1")
def main(ckpt, batch_size, exp_name):
    torch.manual_seed(1234)
    train(ckpt, batch_size, exp_name)


if __name__ == "__main__":
    main()
