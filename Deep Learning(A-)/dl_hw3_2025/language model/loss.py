import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):

    def forward(self, y_hat: torch.Tensor, y: torch.Tensor):
        """
        y_hat: (B, T, vocab_size)
        y: (B, T)
        """
        # Convert y_hat to (B*T, vocab_size), y to (B*T)
        return F.cross_entropy(y_hat.view(-1, y_hat.size(-1)),
                               y.view(-1),
                               ignore_index=-1)


class DPOLoss(nn.Module):

    def __init__(self, kl_beta=0.02):
        super().__init__()
        self.kl_beta = kl_beta

    def forward(self, positive_log_probs, negative_log_probs, positive_log_probs_sft, negative_log_probs_sft):
        """
        positive_log_probs: (B, 1)  -> Policy Model 对 Winner 的 Log Prob
        negative_log_probs: (B, 1)  -> Policy Model 对 Loser 的 Log Prob
        positive_log_probs_sft: (B, 1) -> Reference Model 对 Winner 的 Log Prob
        negative_log_probs_sft: (B, 1) -> Reference Model 对 Loser 的 Log Prob
        """
        ############################ Your code here ############################
        # 1. 计算 Winner 的 Log Ratio (Policy - Reference)
        # 对应公式项: log(p_DPO(yw|x) / p_SFT(yw|x))
        pi_logratios_pos = positive_log_probs - positive_log_probs_sft

        # 2. 计算 Loser 的 Log Ratio (Policy - Reference)
        # 对应公式项: log(p_DPO(yl|x) / p_SFT(yl|x))
        pi_logratios_neg = negative_log_probs - negative_log_probs_sft

        # 3. 计算最终的 Logits (Diff)
        # 对应公式内部: beta * (Ratio_Winner - Ratio_Loser)
        logits = self.kl_beta * (pi_logratios_pos - pi_logratios_neg)

        # 4. 计算 Loss: -log(sigmoid(logits))
        # F.logsigmoid(x) 等价于 log(1 / (1 + exp(-x)))
        loss = -F.logsigmoid(logits).mean()

        return loss
        ########################################################################
