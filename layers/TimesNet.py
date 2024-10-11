import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from peft import get_peft_model, LoraConfig, TaskType

from layers.Conv_Blocks import Inception_Block_V1
from models.GPT2_arch import AccustumGPT2Model


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]

class TimesNet(nn.Module):
    def __init__(self, configs, device, word_embedding,num_heads=8):
        super().__init__()
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=configs.r,
            lora_alpha=configs.lora_alpha,
            lora_dropout=configs.lora_dropout,
            target_modules=["c_attn"]
        )
        self.configs = configs
        self.device = device
        self.seq_len = configs.seq_len
        self.k = 2
        # 划分成几份
        self.patch_k = 3
        self.word_embedding = word_embedding.T
        _, hidden_dim = self.word_embedding.size()
        self.hidden_dim = hidden_dim
        self.cross_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.gpt2 = AccustumGPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2 = get_peft_model(self.gpt2, peft_config)

    def forward(self, x):
        B, T, N = x.size()
        self.n = N
        self.out_channel = 32
        self.conv = nn.Sequential(
            Inception_Block_V1(self.n, self.out_channel,
                               num_kernels=self.configs.num_kernels),
            nn.GELU(),
            Inception_Block_V1(self.out_channel, self.n,
                               num_kernels=self.configs.num_kernels)
        ).to(self.device)
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len) % period != 0:
                length = (
                                 ((self.seq_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            B, N, F, P = out.size()
            # 1. 计算需要填充的长度，使得 P 能被 k 整除
            padding_length = (self.patch_k - (P % self.patch_k)) % self.patch_k  # 计算填充的长度
            if padding_length > 0:
                # 2. 使用 F.pad 对 P 维度进行填充
                # 填充在 P 维度（最后一维），因此在 F.pad 中 paddings 的形式为 (0, padding_length)
                out = F.pad(out, (0, padding_length))  # 只对 P 维度进行填充

            # 填充后新的 P 值
            P = P + padding_length

            # 3. 重新 reshape，将 out 的 P 维度切分为 (P/k, k)
            out = out.view(B, N, F, P // self.patch_k, self.patch_k)

            out = out.permute(0, 1, 3, 2, 4).contiguous()  # 调整维度顺序
            out = out.view(B, N, P * F // self.patch_k, self.patch_k)
            # 对最后一个维度 k 进行平均池化，去掉 k 维度
            out = torch.mean(out, dim=-1)
            B,N, time_dim = out.size()
            # 创建一个线性变换层，将 word_embedding 的维度调整到与 out 的 F*P/K 一致
            linear_proj = nn.Linear(time_dim, self.hidden_dim).to(self.device)
            out = linear_proj(out)

            query = out.permute(1, 0, 2)  # (N, B, hidden_dim)
            # 将 word_embedding 作为键和值，调整形状为 (sequence_length, B, embedding_dim)
            if self.word_embedding.ndim == 2:
                self.word_embedding = self.word_embedding.repeat(B, 1, 1)
            elif self.word_embedding.shape[0] != B:
                self.word_embedding = self.word_embedding[0].repeat(B, 1, 1)
            key_value = self.word_embedding.transpose(0,1)  # (sequence_length, B, embedding_dim)
            # 计算多头注意力，使用 out 作为 query，word_embedding 作为 key 和 value
            attn_output, attn_output_weights = self.cross_attention(query, key_value, key_value)
            # 将多头注意力的输出形状变回 (B, N, hidden_dim)
            attn_output = attn_output.permute(1, 0, 2)  # (B, N, hidden_dim)
            out, intermidiate_feat_time = self.gpt2(inputs_embeds=attn_output)
            res.append(out)  # 保留前 self.seq_len 个时间步的数据
        res = torch.stack(res, dim=-1)
            # 计算加权求和
        # 定义一个可学习的权重，大小为 (num_iterations)
        weights = nn.Parameter(torch.randn(self.k)).to(device=self.device)  # 可学习的权重

        # 对最后一维进行加权求和
        # 首先对权重进行 softmax，以确保加权求和是平滑的
        weights_softmax = nn.functional.softmax(weights, dim=0).to(device=self.device)

        # 对 res 的最后一维进行加权求和，weights_softmax 的形状需要与 res 的最后一维匹配
        res_weighted_sum = torch.sum(res * weights_softmax, dim=-1).to(device=self.device)  # (B, N, dim)

        print("period_size:",res_weighted_sum.shape)  # 输出: (B, N, dim)
        return res_weighted_sum
