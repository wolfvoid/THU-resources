import torch
from torch import nn
from torch.utils.data import DataLoader
from loss import CrossEntropyLoss, DPOLoss
import torch.optim as optim
from torch.cuda.amp.grad_scaler import GradScaler
import statistics
from tqdm import tqdm, trange
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import os
import json
import random
from torchinfo import summary
from configs import TrainingConfig


class Trainer:

    def __init__(self) -> None:
        self.model = None
        self.optimizer = None
        random.seed(1)

    def save_hyperparams(self, hp):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')

        with open(f'./runs/{self.run_name}/hyperparams.json', 'w') as fp:
            json.dump(hp, fp, indent=4)

    def save_metrics(self, metrics):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        with open(f'./runs/{self.run_name}/metrics.json', 'w') as fp:
            json.dump(metrics, fp, indent=4)

    def save_states(self, step, is_last=False):
        if not os.path.exists(f'./runs/{self.run_name}'):
            os.makedirs(f'./runs/{self.run_name}')
        file_name = f'{self.run_name}_final.pt' if is_last else f'{self.run_name}_step{step}.pt'
        torch.save(
            {
                'step': step,
                'model_state_dict':
                    self.model.state_dict(),  # Save the unoptimized model
                'optimizer_state_dict': self.optimizer.state_dict(),
            },
            f'./runs/{self.run_name}/{file_name}')


class SFTTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"sft_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.num_steps = cfg.num_steps
        self.save_freq = 20000
        self.train_dataloader =  iter(
            DataLoader(train_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=8,
                       pin_memory=True))
        self.test_dataloader = iter(
            DataLoader(test_dataset,
                       batch_size=cfg.batch_size,
                       num_workers=8,
                       pin_memory=True))
        self.model = model
        self.criterion = CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def fit(self):
        summary(self.model, input_data=torch.ones(1, 1024).long())

        self.model.to(self.device)
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        self.model.train()

        with trange(1, self.num_steps + 1) as pbar:
            for step in pbar:
                x, y = next(self.train_dataloader)
                x = x.to(self.device)
                y = y.to(self.device)

                with torch.autocast(device_type=self.device, dtype=self.dtype):
                    y_hat = self.model(x)
                    loss = self.criterion(y_hat, y)

                if self.grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.grad_clip)

                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                lossf = loss.item()
                writer.add_scalar('Loss/train/step', lossf, step)

                pbar.set_description(f"Step {step}, batch loss {round(lossf, 3)}")

                if step % self.save_freq == 0:
                    self.save_states(step)

        self.save_states(step, True)


class DPOTrainer(Trainer):

    def __init__(self, cfg: TrainingConfig, device, model: nn.Module, sft_model: nn.Module,
                 train_dataset, test_dataset) -> None:
        super().__init__()
        self.cfg = cfg
        self.run_name = f"dpo_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"
        self.device = device
        assert self.device == 'cuda'
        self.num_epochs = cfg.num_epochs
        self.eval_freq = 5000
        self.save_freq = 20000
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=cfg.batch_size,
                                           num_workers=8,
                                           shuffle=True,
                                           pin_memory=True)
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=cfg.batch_size,
                                          num_workers=8,
                                          pin_memory=True)
        self.model = model
        self.sft_model = sft_model
        self.criterion = DPOLoss(kl_beta=cfg.kl_beta)

        self.optimizer = optim.Adam(self.model.parameters(), lr=cfg.lr)
        self.grad_clip = cfg.grad_clip
        self.dtype = torch.float16

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            **cfg.dict(),
        }
        self.save_hyperparams(hp)

    def shared_step(self, model, sft_model, completions, attention_masks):
        ########################################################################
        # TODO: Implement a single step of DPO trainer
        # TODO: You should implement the model call (including self.model and self.sft_model)
        # TODO: After calling the model and obtaining log_p, calculate the loss and accuracy
        # TODO: Hint: (completions[:, 0], attention_masks[:, 0]) is the positive sample
        #         and (completions[:, 1], attention_masks[:, 1]) is the negative sample
        ############################ Your code here ############################
        # loss, acc = None, None
        # 1. 准备数据：将 completions 和 masks 展平，以便一次性传入模型处理
        # completions shape: (B, 2, T) -> (B * 2, T)
        # 偶数索引 (0, 2...) 为 chosen (winner), 奇数索引 (1, 3...) 为 rejected (loser)
        batch_size = completions.size(0)
        flat_completions = completions.view(batch_size * 2, -1)
        flat_masks = attention_masks.view(batch_size * 2, -1)

        # 2. 计算 Policy Model 的 log probabilities
        # get_log_p 应该返回 (B*2, ) 的 log prob
        policy_log_ps = self.model.get_log_p(flat_completions, flat_masks)

        # 3. 计算 Reference (SFT) Model 的 log probabilities
        # Reference model 不需要梯度，使用 no_grad 节省显存和计算
        with torch.no_grad():
            ref_log_ps = self.sft_model.get_log_p(flat_completions, flat_masks)

        # 4. 将结果拆分回 chosen 和 rejected
        # view 回 (B, 2)
        policy_log_ps = policy_log_ps.view(batch_size, 2)
        ref_log_ps = ref_log_ps.view(batch_size, 2)

        policy_chosen_logps = policy_log_ps[:, 0]
        policy_rejected_logps = policy_log_ps[:, 1]

        ref_chosen_logps = ref_log_ps[:, 0]
        ref_rejected_logps = ref_log_ps[:, 1]

        # 5. 计算 Loss
        # criterion (DPOLoss) 通常接受 (policy_chosen, policy_rejected, ref_chosen, ref_rejected)
        loss = self.criterion(policy_chosen_logps, policy_rejected_logps,
                              ref_chosen_logps, ref_rejected_logps)

        # 6. 计算 Accuracy
        # 准确率定义为：Policy 认为 Chosen 比 Rejected 更好的比例
        # 也就是检查隐式奖励: r(x, y) = beta * (log_policy - log_ref)
        # 判断条件: (log_policy_chosen - log_ref_chosen) > (log_policy_rejected - log_ref_rejected)
        chosen_rewards = policy_chosen_logps - ref_chosen_logps
        rejected_rewards = policy_rejected_logps - ref_rejected_logps

        # 转换为 float 计算均值
        acc = (chosen_rewards > rejected_rewards).float().mean()
        ########################################################################
        return loss, acc

    def fit(self):
        summary(self.model, input_data=torch.ones(1, 1024).long())

        self.model.to(self.device)
        self.sft_model.to(self.device)
        self.sft_model.eval()
        writer = SummaryWriter(f'./runs/{self.run_name}/logs', max_queue=40)
        scaler = GradScaler(enabled=self.dtype != torch.float32)

        self.model.train()

        steps = 0

        for epoch in range(1, self.num_epochs + 1):
            with tqdm(self.train_dataloader) as pbar:
                for completions, attention_masks in pbar:
                    steps += 1
                    completions = completions.to(self.device)
                    attention_masks = attention_masks.to(self.device)

                    with torch.autocast(device_type=self.device, dtype=self.dtype):
                        loss, acc = self.shared_step(self.model, self.sft_model, completions, attention_masks)

                    if self.grad_clip != 0.0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                    self.grad_clip)

                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    lossf = loss.item()
                    accf = acc.item()
                    writer.add_scalar('Loss/train/step', lossf, steps)
                    writer.add_scalar('Acc/train/step', accf, steps)
                    pbar.set_description(f"Epoch {epoch}, batch loss {round(lossf, 3)}, acc {round(accf, 3)}")

                    if steps != 0 and steps % self.save_freq == 0:
                        self.save_states(steps)

                    if steps % self.eval_freq == 0:
                        with torch.no_grad():
                            self.model.eval()
                            losses, accs = [], []
                            for completions, attention_masks in tqdm(self.test_dataloader):
                                completions = completions.to(self.device)
                                attention_masks = attention_masks.to(self.device)
                                loss, acc = self.shared_step(self.model, self.sft_model, completions, attention_masks)
                                lossf, accf = loss.item(), acc.item()
                                losses.append(lossf)
                                accs.append(accf)
                            total_loss, total_acc = statistics.mean(losses), statistics.mean(accs)
                            writer.add_scalar('Loss/test/step', total_loss, steps)
                            writer.add_scalar('Acc/test/step', total_acc, steps)
                            print(f'Step: {steps + 1}, Test Loss: {total_loss}, Acc: {total_acc}')

            self.save_states(steps, True)



## TODO LORA
class LORATrainer(SFTTrainer):

    def __init__(self, cfg, device, model, train_dataset, test_dataset) -> None:
        # 1. 调用父类初始化
        # 注意：SFTTrainer.__init__ 会初始化一个包含所有参数的 optimizer
        super().__init__(cfg, device, model, train_dataset, test_dataset)

        # 2. 修改 Run Name 以便于区分
        self.run_name = f"lora_{cfg.exp_name}_{datetime.now().strftime('%Y%m%d%H%M')}"

        # 3. 重新初始化 Optimizer
        # LoRA 的核心优势是省显存。如果使用 SFTTrainer 原本的 optimizer，
        # 它会为所有参数（包括冻结的）维护 momentum 状态，这会抵消 LoRA 的显存优势。
        # 我们这里覆盖 self.optimizer，只传入 requires_grad=True 的参数。
        self.optimizer = optim.Adam(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=cfg.lr
        )

        # 4. 更新保存的超参数信息
        # 记录一下实际训练的参数量，方便在 Tensorboard/JSON 中查看
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        hp = {
            "dtype": str(self.dtype),
            "train_dataset": type(train_dataset).__name__,
            "train_dataset_len": len(train_dataset),
            "test_dataset": type(test_dataset).__name__,
            "test_dataset_len": len(test_dataset),
            "is_lora": True,
            "trainable_params": trainable_params,
            **cfg.dict(),
        }
        self.save_hyperparams(hp)
