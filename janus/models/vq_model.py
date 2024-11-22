# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#这段代码定义了一个包含**编码器（Encoder）和解码器（Decoder）**的神经网络架构，主要用于图像处理任务。网络由多个模块组成，包括残差块（ResNet Block）和注意力块（AttnBlock）。

from dataclasses import dataclass, field
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

"""这个类定义了一些模型的超参数
codebook_size: 用于量化嵌入的码本大小。
codebook_embed_dim: 码本中每个向量的维度。
encoder_ch_mult 和 decoder_ch_mult: 控制通道数在不同分辨率层级上的增长。
z_channels: 表示潜在特征（latent features）的通道数"""
@dataclass
class ModelArgs:
    codebook_size: int = 16384
    codebook_embed_dim: int = 8
    codebook_l2_norm: bool = True #l2_归一化
    codebook_show_usage: bool = True #显示代码本的使用情况
    commit_loss_beta: float = 0.25 #用来控制提交损失（commitment loss）的强度
    entropy_loss_ratio: float = 0.0 #熵损失通常用于鼓励模型的输出具有高熵（即更加分散），但在这个设置中它被设置为 0.0，表示不使用熵损失

    encoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])#用于表示编码器每一层的通道数倍率，表示随着网络的深入，通道数会逐渐增加。
    decoder_ch_mult: List[int] = field(default_factory=lambda: [1, 1, 2, 2, 4])
    z_channels: int = 256 #表示模型中 z 向量的通道数。z 向量通常用于表示潜在空间的特征，设置为 256
    dropout_p: float = 0.0#表示 Dropout 正则化的概率，通常用于防止过拟合。0.0 表示不使用 Dropout


class Encoder(nn.Module):
    #初始化了编码器模型的各个层
    def __init__(
        self,
        in_channels=3,#输入图像通道数
        ch=128,#基础通道数
        ch_mult=(1, 1, 2, 2, 4),#控制每层通道数
        num_res_blocks=2,#每个分辨率层级包含的残差块数量
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,#使用卷积进行下采样
        z_channels=256,输出潜在特征通道数
    ):
        super().__init__()#调用父类 nn.Module 的初始化函数，初始化父类的所有内容。
        self.num_resolutions = len(ch_mult)#分辨率层数
        self.num_res_blocks = num_res_blocks#每个分辨率层级的残差块数
        self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, stride=1, padding=1)#定义输入卷积层

        # downsampling
        in_ch_mult = (1,) + tuple(ch_mult)#生成一个新的通道数倍增序列，前面加了一个 1，确保第一个分辨率层的输入通道数是 1, (1,1,1,2,2,4)
        self.conv_blocks = nn.ModuleList()#用于存储每一层的卷积块
        for i_level in range(self.num_resolutions):#遍历每一层
            conv_block = nn.Module()#为每个分辨率级别创建一个空的模块（conv_block），它将在后面被加入包含残差块和注意力块。
            # res & attn
            res_block = nn.ModuleList()#用于储存残差块
            attn_block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level] #根据当前层级 i_level 和基础通道数配置ch，计算该层的输入通道数（block_in）和输出通道数（block_out）如i_level=2,128*1=128
            block_out = ch * ch_mult[i_level] # i_level=2, 128*2=256
            for _ in range(self.num_res_blocks):#在每个分辨率层级中，添加 num_res_blocks 个残差块（ResnetBlock
                res_block.append(
                    ResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out #更新
                if i_level == self.num_resolutions - 1:#在最后一个分辨率层级（最小分辨率）后，添加一个注意力块（AttnBlock）
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block#将残差块和注意力块添加到 conv_block 中
            conv_block.attn = attn_block
            # downsample
            if i_level != self.num_resolutions - 1: #如果当前层不是最后一层，则进行下采样，将特征图大小减小
                conv_block.downsample = Downsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)#将当前的卷积块（conv_block）添加到 self.conv_blocks 列表中。

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(
            block_in, z_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        h = self.conv_in(x)
        # downsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.downsample(h)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        z_channels=256,
        ch=128,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        norm_type="group",
        dropout=0.0,
        resamp_with_conv=True,
        out_channels=3,
    ):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        block_in = ch * ch_mult[self.num_resolutions - 1]
        # z to block_in
        self.conv_in = nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.ModuleList()
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )
        self.mid.append(AttnBlock(block_in, norm_type=norm_type))
        self.mid.append(
            ResnetBlock(block_in, block_in, dropout=dropout, norm_type=norm_type)
        )

        # upsampling
        self.conv_blocks = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            conv_block = nn.Module()
            # res & attn
            res_block = nn.ModuleList()
            attn_block = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                res_block.append(
                    ResnetBlock(
                        block_in, block_out, dropout=dropout, norm_type=norm_type
                    )
                )
                block_in = block_out
                if i_level == self.num_resolutions - 1:
                    attn_block.append(AttnBlock(block_in, norm_type))
            conv_block.res = res_block
            conv_block.attn = attn_block
            # downsample
            if i_level != 0:
                conv_block.upsample = Upsample(block_in, resamp_with_conv)
            self.conv_blocks.append(conv_block)

        # end
        self.norm_out = Normalize(block_in, norm_type)
        self.conv_out = nn.Conv2d(
            block_in, out_channels, kernel_size=3, stride=1, padding=1
        )

    @property
    def last_layer(self):
        return self.conv_out.weight

    def forward(self, z):
        # z to block_in
        h = self.conv_in(z)

        # middle
        for mid_block in self.mid:
            h = mid_block(h)

        # upsampling
        for i_level, block in enumerate(self.conv_blocks):
            for i_block in range(self.num_res_blocks + 1):
                h = block.res[i_block](h)
                if len(block.attn) > 0:
                    h = block.attn[i_block](h)
            if i_level != self.num_resolutions - 1:
                h = block.upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h

#实现向量量化（Vector Quantization），用于将连续潜在空间的特征映射到离散码本（codebook）中的索引。
class VectorQuantizer(nn.Module):
    def __init__(self, n_e, e_dim, beta, entropy_loss_ratio, l2_norm, show_usage):
        super().__init__()
        self.n_e = n_e#码本大小（即离散向量的数量）
        self.e_dim = e_dim#每个码本向量的维度
        self.beta = beta#权重
        self.entropy_loss_ratio = entropy_loss_ratio
        self.l2_norm = l2_norm
        self.show_usage = show_usage#是否统计码本的使用情况

        """使用 nn.Embedding 创建一个大小为 (n_e, e_dim) 的嵌入矩阵，表示码本。
        使用随机均匀分布初始化码本向量。
        如果启用了 L2 归一化，则对码本进行归一化处理"""
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.n_e, 1.0 / self.n_e)
        if self.l2_norm:
            self.embedding.weight.data = F.normalize(
                self.embedding.weight.data, p=2, dim=-1
            )
        #如果启用了 show_usage，用于记录码本中每个向量的使用次数。
        if self.show_usage:
            self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = torch.einsum("b c h w -> b h w c", z).contiguous()
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        if self.l2_norm:
            z = F.normalize(z, p=2, dim=-1)
            z_flattened = F.normalize(z_flattened, p=2, dim=-1)
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight

        d = (
            torch.sum(z_flattened**2, dim=1, keepdim=True)
            + torch.sum(embedding**2, dim=1)
            - 2
            * torch.einsum(
                "bd,dn->bn", z_flattened, torch.einsum("n d -> d n", embedding)
            )
        )

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = embedding[min_encoding_indices].view(z.shape)
        perplexity = None
        min_encodings = None
        vq_loss = None
        commit_loss = None
        entropy_loss = None

        # compute loss for embedding
        if self.training:
            vq_loss = torch.mean((z_q - z.detach()) ** 2)#量化损失
            commit_loss = self.beta * torch.mean((z_q.detach() - z) ** 2)
            entropy_loss = self.entropy_loss_ratio * compute_entropy_loss(-d)

        # preserve gradients 梯度反向传播修正
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = torch.einsum("b h w c -> b c h w", z_q)

        return (
            z_q,
            (vq_loss, commit_loss, entropy_loss),
            (perplexity, min_encodings, min_encoding_indices),
        )

    def get_codebook_entry(self, indices, shape=None, channel_first=True):
        # shape = (batch, channel, height, width) if channel_first else (batch, height, width, channel)
        if self.l2_norm:
            embedding = F.normalize(self.embedding.weight, p=2, dim=-1)
        else:
            embedding = self.embedding.weight
        z_q = embedding[indices]  # (b*h*w, c)

        if shape is not None:
            if channel_first:
                z_q = z_q.reshape(shape[0], shape[2], shape[3], shape[1])
                # reshape back to match original input shape
                z_q = z_q.permute(0, 3, 1, 2).contiguous()
            else:
                z_q = z_q.view(shape)
        return z_q

"""实现了标准的残差块结构，用于增强模型的特征提取能力。
在输入和输出之间添加了残差连接（Residual Connection），缓解梯度消失问题"""
class ResnetBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout=0.0,
        norm_type="group",
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels, norm_type)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.norm2 = Normalize(out_channels, norm_type)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

        #残差连接：如果输入和输出通道不同，通过卷积调整输入的形状。
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    #特征经过两个卷积和激活层处理，最后与输入 x 相加形成残差输出。
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h

#实现了基于点的自注意力机制，捕获全局的特征依赖关系。
class AttnBlock(nn.Module):
    def __init__(self, in_channels, norm_type="group"):
        super().__init__()
        self.norm = Normalize(in_channels, norm_type)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )#使用 proj_out 将注意力结果映射回原始特征空间。

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = F.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_

#使用 Swish 激活函数
def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


#提供两种规范化方式（Group Norm 和 Batch Norm），用于稳定训练。
def Normalize(in_channels, norm_type="group"):
    assert norm_type in ["group", "batch"]
    if norm_type == "group":
        return nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
        )
    elif norm_type == "batch":
        return nn.SyncBatchNorm(in_channels)

"""对输入特征图进行上采样操作，将其分辨率提高一倍。
可选地在上采样后添加卷积层，进一步处理特征"""
class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv#布尔值，表示是否在上采样后添加卷积层
        if self.with_conv:#如果真，定义一个卷积层
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        """上采样：使用最近邻插值（mode="nearest"）将特征图分辨率放大 2 倍。
        如果输入张量的类型不是 float32，会在插值前将其转换为 float32，插值后再转换回原类型。"""
        if x.dtype != torch.float32:
            x = F.interpolate(x.to(torch.float), scale_factor=2.0, mode="nearest").to(
                torch.bfloat16
            )
        else:
            x = F.interpolate(x, scale_factor=2.0, mode="nearest")

        if self.with_conv:
            x = self.conv(x)
        return x


"""对输入特征图进行下采样操作，将其分辨率降低一半。
可选地在下采样时使用卷积层，提取低分辨率特征。"""
class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:#如果使用卷积下采样：由于 PyTorch 不支持非对称填充，需要手动对输入张量填充：
            pad = (0, 1, 0, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:#如果不使用卷积：使用平均池化实现下采样
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
        return x

#计算熵损失，用于鼓励码本向量的均匀使用。熵损失通过计算样本熵和平均熵的差值实现。
def compute_entropy_loss(affinity, loss_type="softmax", temperature=0.01):
    """将输入的相似性矩阵展平：affinity 通常是特征与码本的相似性分数。
        可通过温度参数 temperature 调整分数分布。"""
    flat_affinity = affinity.reshape(-1, affinity.shape[-1])
    flat_affinity /= temperature
    #计算概率分布
    probs = F.softmax(flat_affinity, dim=-1)
    log_probs = F.log_softmax(flat_affinity + 1e-5, dim=-1)
    
    if loss_type == "softmax":
        target_probs = probs
    else:
        raise ValueError("Entropy loss {} not supported".format(loss_type))
    avg_probs = torch.mean(target_probs, dim=0)
    avg_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-5))
    sample_entropy = -torch.mean(torch.sum(target_probs * log_probs, dim=-1))
    #返回熵损失
    loss = sample_entropy - avg_entropy
    return loss

#实现了一个完整的向量量化模型，包括编码器、解码器和向量量化模块。
class VQModel(nn.Module):
    #使用 config 配置模型的超参数，包括编码器/解码器的通道倍增系数、码本大小、损失权重等。
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        #编码器将输入映射到潜在空间。
        self.encoder = Encoder(
            ch_mult=config.encoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )
        #解码器将潜在空间的特征恢复为原始分辨率
        self.decoder = Decoder(
            ch_mult=config.decoder_ch_mult,
            z_channels=config.z_channels,
            dropout=config.dropout_p,
        )

        self.quantize = VectorQuantizer(
            config.codebook_size,
            config.codebook_embed_dim,
            config.commit_loss_beta,
            config.entropy_loss_ratio,
            config.codebook_l2_norm,
            config.codebook_show_usage,
        )
        #用于调整特征的通道数
        self.quant_conv = nn.Conv2d(config.z_channels, config.codebook_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(
            config.codebook_embed_dim, config.z_channels, 1
        )

    #提取输入的潜在特征，并进行量化。
    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info

    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec

    def decode_code(self, code_b, shape=None, channel_first=True):
        quant_b = self.quantize.get_codebook_entry(code_b, shape, channel_first)
        dec = self.decode(quant_b)
        return dec

    def forward(self, input):
        quant, diff, _ = self.encode(input)
        dec = self.decode(quant)
        return dec, diff


#################################################################################
#                              VQ Model Configs                                 #
#################################################################################
"""模型配置函数
VQ_16 函数
预设了编码器和解码器的通道倍增系数，快速创建一个 VQ 模型："""
def VQ_16(**kwargs):
    return VQModel(
        ModelArgs(
            encoder_ch_mult=[1, 1, 2, 2, 4], decoder_ch_mult=[1, 1, 2, 2, 4], **kwargs
        )
    )


VQ_models = {"VQ-16": VQ_16}
