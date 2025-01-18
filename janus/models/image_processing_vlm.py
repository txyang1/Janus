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
#第二步
#这段代码定义了一个 VLMImageProcessor 类，用于深度学习视觉模型的图像预处理。它支持图像的大小调整、归一化、标准化等操作，并且专为多模态学习任务优化。
from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision
import torchvision.transforms.functional
from PIL import Image
from transformers import AutoImageProcessor, PretrainedConfig
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import to_numpy_array
from transformers.utils import logging

logger = logging.get_logger(__name__) #设置日志工具：用于记录和调试图像处理器的运行信息。


#类型和标准值
ImageType = Union[np.ndarray, torch.Tensor, Image.Image]
IMAGENET_MEAN = (0.48145466, 0.4578275, 0.40821073)
IMAGENET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)
"""图像resize：通过如下计算方法，将图像等比例的进行缩放，
令图像的高或宽（较大的那个）变为self.image_size大小，防止图像畸形，
然后调用expand2square方法用背景色将图像填充成self.image_size*self.image_size（1024*1024）大小。"""

def expand2square(pil_img, background_color):#定义一个函数，将非正方形图像扩展为正方形。
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)#创建一个宽度为 width 的正方形图像，填充背景颜色
        result.paste(pil_img, (0, (width - height) // 2))#将原始图像粘贴到新图像的中间。
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


class VLMImageProcessorConfig(PretrainedConfig):#定义一个配置类，用于存储图像处理的参数。
    model_type = "deepseek_vlm"#定义模型类型标识符。
    #定义图像处理所需的参数字段。
    image_size: int
    min_size: int
    image_mean: Union[Tuple[float, float, float], List[float]]
    image_std: Union[Tuple[float, float, float], List[float]]
    rescale_factor: float
    do_normalize: bool

    #初始化方法：接收并设置图像处理器的配置参数，提供默认值。
    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,#它用于确保类能够扩展和继承父类的所有参数，而不丢失功能。
    ):
        #将参数保存为类实例的属性。
        self.image_size = image_size
        self.min_size = min_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize

        super().__init__(**kwargs)#调用父类 PretrainedConfig 的初始化方法。


class VLMImageProcessor(BaseImageProcessor):#定义自定义图像处理器类，继承自 BaseImageProcessor
    model_input_names = ["pixel_values"]#定义模型的输入名称。

    #初始化方法：接收图像处理器的参数。
    def __init__(
        self,
        image_size: int,
        min_size: int = 14,
        image_mean: Union[Tuple[float, float, float], List[float]] = (
            0.48145466,
            0.4578275,
            0.40821073,
        ),
        image_std: Union[Tuple[float, float, float], List[float]] = (
            0.26862954,
            0.26130258,
            0.27577711,
        ),
        rescale_factor: float = 1.0 / 255.0,
        do_normalize: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)#调用父类 BaseImageProcessor 的初始化方法。

        #将参数保存为实例属性。
        self.image_size = image_size
        self.rescale_factor = rescale_factor
        self.image_mean = image_mean
        self.image_std = image_std
        self.min_size = min_size
        self.do_normalize = do_normalize

        #如果未设置 image_mean，背景颜色默认为中性灰色，否则根据 image_mean 计算背景颜色。
        if image_mean is None:
            self.background_color = (127, 127, 127)
        else:
            self.background_color = tuple([int(x * 255) for x in image_mean])#遍历 image_mean 中的每个通道的均值 x，将其乘以 255，将浮点数扩展到整数范围

    #定义图像调整大小的核心方法，将输入图像调整到目标尺寸，并填充为正方形
    def resize(self, pil_img: Image) -> np.ndarray:
        """

        Args:
            pil_img (PIL.Image): [H, W, 3] in PIL.Image in RGB

        Returns:
            x (np.ndarray): [3, self.image_size, self.image_size]
        """

        """像素值归一化：原始像素值为0-255，转换为0-1。
像素值标准化：均值为[0.48145466,0.4578275,0.40821073],方差为[0.26862954,0.26130258,0.27577711]"""
        #获取图像尺寸，计算最大边长。
        width, height = pil_img.size
        max_size = max(width, height)

        #根据 image_size 和 min_size 计算调整后的尺寸。
        size = [
            max(int(height / max_size * self.image_size), self.min_size),
            max(int(width / max_size * self.image_size), self.min_size),
        ]

        #检查输入图像尺寸是否有效。
        if width <= 0 or height <= 0 or size[0] <= 0 or size[1] <= 0:
            print(f"orig size = {pil_img.size}, new size = {size}")
            raise ValueError("Invalid size!")

        #使用双三次插值（Bicubic Interpolation）调整图像大小。
        pil_img = torchvision.transforms.functional.resize(
            pil_img,
            size,
            interpolation=torchvision.transforms.functional.InterpolationMode.BICUBIC,
            antialias=True,
        )

        #调用 expand2square 填充图像为正方形，并转换为 NumPy 数组。
        pil_img = expand2square(pil_img, self.background_color)
        x = to_numpy_array(pil_img)

        # [H, W, 3] -> [3, H, W]
        x = np.transpose(x, (2, 0, 1))

        return x

    #定义预处理方法，接收图像列表并返回标准化后的数据。
    def preprocess(self, images, return_tensors: str = "pt", **kwargs) -> BatchFeature:
        # resize and pad to [self.image_size, self.image_size]
        # then convert from [H, W, 3] to [3, H, W]
        images: List[np.ndarray] = [self.resize(image) for image in images]#对输入图像逐个调用 resize 方法。

        # resacle from [0, 255] -> [0, 1] 调用 rescale 方法，将像素值从 [0, 255] 缩放到 [0, 1]
        images = [
            self.rescale(
                image=image,
                scale=self.rescale_factor,
                input_data_format="channels_first",
            )
            for image in images
        ]

        # normalize 如果 do_normalize 为真，则对图像进行标准化处理。
        if self.do_normalize:
            images = [
                self.normalize(
                    image=image,
                    mean=self.image_mean,
                    std=self.image_std,
                    input_data_format="channels_first",
                )
                for image in images
            ]

        #将处理后的图像包装成 BatchFeature 对象返回。
        data = {"pixel_values": images}
        return BatchFeature(data=data, tensor_type=return_tensors)

    @property
    def default_shape(self):
        return [3, self.image_size, self.image_size]

#注册自定义图像处理器，使其可通过 AutoImageProcessor 加载。
AutoImageProcessor.register(VLMImageProcessorConfig, VLMImageProcessor)

#定义程序的入口点。
if __name__ == "__main__":
    #创建一个图像处理器实例，用于测试配置和逻辑。
    image_processor = VLMImageProcessor(
        image_size=1024,
        image_mean=IMAGENET_INCEPTION_MEAN,
        image_std=IMAGENET_INCEPTION_STD,
        do_normalize=True,
    )
