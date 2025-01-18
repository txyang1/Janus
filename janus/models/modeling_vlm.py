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
#这段代码定义了一个 多模态因果语言模型 (MultiModalityCausalLM)，允许将视觉和语言信息集成到单一模型中。
#它使用了 HuggingFace Transformers 框架并通过扩展其 PreTrainedModel 和 AutoConfig 提供了多模态支持
import torch
from attrdict import AttrDict#将字典的键作为属性访问的工具
from einops import rearrange#用于高效处理张量形状变换。
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
#从janus.models导入自定义模型组件
from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector


class vision_head(torch.nn.Module):#用于处理视觉特征，将输入嵌入处理成视觉特定的表示。
    def __init__(self, params):
        super().__init__()#初始化一个多层感知机 (MLP)
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )#全连接层，将嵌入维度从 n_embed 转换为 image_token_embed
        self.vision_activation = torch.nn.GELU()#激活函数
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )#另一个全连接层，将维度从 image_token_embed 转换为 image_token_size，生成最终的视觉token表示

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x

#动态类加载器
def model_name_to_cls(cls_name):#接收一个类名字符串 cls_name，根据字符串内容动态加载对应的类
    if "MlpProjector" in cls_name:#如果类名字符串中包含 "MlpProjector"，则返回 MlpProjector 类
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls

#配置类定义
class VisionConfig(PretrainedConfig):#定义视觉模型的配置，用于初始化视觉相关模块，存储和管理初始化视觉相关模块所需的参数，包括模型类型、类名和超参数
    model_type = "vision"#表示模型类型
    cls: str = ""#保存视觉模块类名
    params: AttrDict = {}#保存与视觉模块相关的参数

    def __init__(self, **kwargs):#通过 **kwargs 接收任意数量的关键字参数
        super().__init__(**kwargs)#调用父类 PretrainedConfig 的构造函数，初始化基础配置

        self.cls = kwargs.get("cls", "")#从 kwargs 中提取 "cls" 参数，如果没有提供 "cls" 参数，则默认设置为空字符串 ""
        if not isinstance(self.cls, str):#如果 cls 不是字符串（可能是一个类对象），通过 __name__ 获取其类名并将其转换为字符串。
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))#将提取的参数封装为 AttrDict 对象，使其支持点操作访问属性。例如，可以通过 self.params.some_key 访问


class AlignerConfig(PretrainedConfig):#对齐模型配置
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):#对应生成视觉模型配置
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):#对应生成对齐模型配置
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):#对应生成头部模型配置
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))

#多模态配置类，用于管理视觉、对齐、生成（vision、aligner）、以及语言处理（如 Llama 模型）的设置
class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig
    #生成模块配置
    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig
    #语言模型配置（Llama）
    language_config: LlamaConfig

    def __init__(self, **kwargs):#从字典中提取配置，实例化对应的子配置类。
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})#从传入的参数 kwargs 中获取 vision_config 配置字典。如果没有提供，则默认是一个空字典 {}
        self.vision_config = VisionConfig(**vision_config)#使用 VisionConfig(**vision_config) 来实例化一个 VisionConfig 对象，赋值给 self.vision_config

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)

#定义预训练模型基类
class MultiModalityPreTrainedModel(PreTrainedModel):#继承 PreTrainedModel，设定默认配置和基本参数
    config_class = MultiModalityConfig#使用 MultiModalityConfig 类来加载和管理模型配置
    base_model_prefix = "multi_modality"
    _no_split_modules = []#用于指定哪些模块在模型切分（例如分布式训练时）时不应该被拆分的列表
    _skip_keys_device_placement = "past_key_values"#指示在某些操作中跳过设备放置（例如，模型的输入或输出张量的设备分配）时的特定键

#多模态因果语言模型
class MultiModalityCausalLM(MultiModalityPreTrainedModel):#该类整合了语言模型、视觉模型和对齐模块，能够同时处理文本和视觉输入，并生成统一的嵌入。
    def __init__(self, config: MultiModalityConfig):##初始化语言模型：从配置中提取各模块的配置，动态加载对应的模型类
        super().__init__(config)
        #加载视觉模型：根据 vision_config.cls 动态加载视觉模型类并实例化
        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)
        #加载对齐模块：同样，根据配置动态加载
        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        #加载生成模块
        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        #定义生成模块的嵌入层
        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        #加载语言模型
        language_config = config.language_config
        self.language_model = LlamaForCausalLM(language_config)
    #输入准备，准备输入嵌入，定义方法，将文本和图像特征组合成统一的嵌入。 DeepSeek-VL设计了prepare_inputs_embeds去统一提取文本和图像的embedding
    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """
        
        #处理图像嵌入：通过视觉模型提取特征，然后通过对齐模块生成嵌入。
        bs, n = pixel_values.shape[0:2]#提取前两维 batchsize,n_images
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")#将批量维度和图像数量维度合并，以方便后续的图像批量处理
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))
        
        #调整图像嵌入形状：将嵌入调整为模型输入需要的形状。
        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        #对文本输入进行预处理
        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        #将图像嵌入替换到文本嵌入中，完成多模态输入的整合
        # replace with the image embeddings 
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds

    # 图像生成嵌入，根据图像标识 (image_ids)，生成用于生成的图像嵌入
    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))

#注册配置和模型：将自定义的配置类和模型类注册到 transformers 框架中，方便后续加载。
AutoConfig.register("vision", VisionConfig)#将 config_name 和自定义的配置类关联起来。
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)
