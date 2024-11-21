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
#这段代码实现了一个通用的多层感知器（MLP）投影器类 MlpProjector，可以根据配置参数执行不同类型的投影操作。
from typing import Tuple, Union

import torch
import torch.nn as nn
from attrdict import AttrDict


class MlpProjector(nn.Module):#继承自 nn.Module 的多层感知机投影器类
    def __init__(self, cfg): #cfg：配置对象，包含网络的超参数。
        super().__init__()

        #保存配置和初始化模块
        self.cfg = cfg

        if cfg.projector_type == "identity":
            modules = nn.Identity()

        elif cfg.projector_type == "linear":
            modules = nn.Linear(cfg.input_dim, cfg.n_embed)

        elif cfg.projector_type == "mlp_gelu":#如果 projector_type 是 "mlp_gelu"：从配置中获取多层感知机的深度，默认值为 
            mlp_depth = cfg.get("depth", 1)
            #初始化第一层线性投影，将输入维度映射到嵌入维度。
            modules = [nn.Linear(cfg.input_dim, cfg.n_embed)]
            #添加多层感知机的后续层：每层包含 GELU 激活函数和一个线性投影层。
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            #使用 nn.Sequential 将层列表打包成一个顺序模型
            modules = nn.Sequential(*modules)

        #特殊的低高分辨率混合 MLP
        #如果 projector_type 是 "low_high_hybrid_split_mlp_gelu"：获取多层感知机的深度
        elif cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            mlp_depth = cfg.get("depth", 1)
            #定义两条独立的线性投影路径：high_up_proj 处理高分辨率输入。low_up_proj 处理低分辨率输入。两者的输出维度均为嵌入维度的一半。
            self.high_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)
            self.low_up_proj = nn.Linear(cfg.input_dim, cfg.n_embed // 2)

            #后续层的设置与 mlp_gelu 类似，每层包含 GELU 激活函数和线性投影
            modules = []
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(cfg.n_embed, cfg.n_embed))
            #将这些层封装成顺序模型。
            modules = nn.Sequential(*modules)

        else:
            raise ValueError(f"Unknown projector type: {cfg.projector_type}")

        #保存网络层,将构建好的模块保存为类的属性
        self.layers = modules

    def forward(
        self, x_or_tuple: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]
    ):
        """

        输入参数：x_or_tuple：
        如果是 torch.Tensor：单一特征输入。
        如果是元组：来自混合视觉编码器的高分辨率和低分辨率特征。
        Args:
            x_or_tuple (Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:  if it is a tuple of torch.Tensor,
                then it comes from the hybrid vision encoder, and x = high_res_x, low_res_x);
                otherwise it is the feature from the single vision encoder.

        Returns:
            x (torch.Tensor): [b, s, c]
        """

        #如果输入是元组，拆分为 high_x（高分辨率特征）和 low_x（低分辨率特征）
        if isinstance(x_or_tuple, tuple):
            # self.cfg.projector_type == "low_high_hybrid_split_mlp_gelu":
            high_x, low_x = x_or_tuple
            #分别对高低分辨率特征进行线性投影，然后在最后一维上拼接
            high_x = self.high_up_proj(high_x)
            low_x = self.low_up_proj(low_x)
            x = torch.concat([high_x, low_x], dim=-1)
        else: #如果输入是单一张量，则直接赋值。
            x = x_or_tuple

        #将输入 x 传递给构建好的网络层并返回结果。
        return self.layers(x)

#测试代码 定义配置
if __name__ == "__main__":
    cfg = AttrDict(
        input_dim=1024,
        n_embed=2048,
        depth=2,
        projector_type="low_high_hybrid_split_mlp_gelu",
    )
    #生成输入
    inputs = (torch.rand(4, 576, 1024), torch.rand(4, 576, 1024))

    #实例化模型并执行
    m = MlpProjector(cfg)
    out = m(inputs)
    print(out.shape)
