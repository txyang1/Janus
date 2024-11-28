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
#这段代码定义了一个名为 VLChatProcessor 的类，专门用于对多模态输入（文本和图像）进行预处理，以便用于视觉-语言模型（如基于 LLaMA 的模型）。
from dataclasses import dataclass #用于定义数据类，便于存储结构化数据。
from typing import Dict, List

import torch
from PIL.Image import Image
from transformers import LlamaTokenizerFast #Hugging Face提供的快速分词器。
from transformers.processing_utils import ProcessorMixin #Hugging Face的工具，用于自定义处理器

from janus.models.image_processing_vlm import VLMImageProcessor #自定义的视觉语言模型（VLM）图像处理模块。
from janus.utils.conversation import get_conv_template #获取对话模板的工具函数。

#用于封装输出的类。
class DictOutput(object):
    def keys(self):
        return self.__dict__.keys()

    def __getitem__(self, item):#支持通过键访问属性值（dict[key]）
        return self.__dict__[item]

    def __setitem__(self, key, value):#支持通过键设置属性值（dict[key] = value）
        self.__dict__[key] = value


@dataclass #数据类，是 Python 的一个装饰器，用于简化类的定义。
class VLChatProcessorOutput(DictOutput):#表示单个数据实例的处理结果，继承自DictOutput
    sft_format: str #用于生成对话的格式
    input_ids: torch.Tensor #文本输入的张量。
    pixel_values: torch.Tensor #图像的像素张量。
    num_image_tokens: torch.IntTensor #图像对应的标记数量

    def __len__(self):
        return len(self.input_ids)#定义了类的 __len__ 方法，允许直接使用 len() 函数获取对象的长度


@dataclass
class BatchedVLChatProcessorOutput(DictOutput):#表示批量多模态数据的处理结果
    sft_format: List[str]
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_emb_mask: torch.BoolTensor

    def to(self, device, dtype=torch.bfloat16):#将所有张量移动到指定设备（如 GPU）并调整数据类型
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)#将图像像素张量移动到指定设备，并转换为指定的数据类型（默认是 torch.bfloat16，通常用于节省显存和提高性能）
        return self

#主处理器类
class VLChatProcessor(ProcessorMixin): #该类继承自 ProcessorMixin。这个类的主要目的是提供处理多模态输入（文本 + 图像）的功能，同时定义了一些默认的静态属性以配置类的行为和默认行为
    image_processor_class = "AutoImageProcessor" #指定默认图像处理类
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast") #支持的分词器类型

    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )#默认的系统提示

    #构造方法：初始化主要组件（图像处理器和分词器）和相关配置。
    #它同时对分词器的特殊标记进行动态扩展，以支持特定的多模态输入格式。
    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",#用于表示图像位置的特殊标记，在文本中占位
        image_start_tag: str = "<begin_of_image>",
        image_end_tag: str = "<end_of_image>",
        num_image_tokens: int = 576,
        add_special_token: bool = False,
        sft_format: str = "deepseek",
        mask_prompt: bool = True,
        ignore_id: int = -100,
        **kwargs,
    ):
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        #动态添加特殊标记
        image_id = self.tokenizer.vocab.get(image_tag)
        if image_id is None:
            special_tokens = [image_tag]
            special_tokens_dict = {"additional_special_tokens": special_tokens}
            self.tokenizer.add_special_tokens(special_tokens_dict)
            print(f"Add image tag = {image_tag} to the tokenizer")

        self.image_tag = image_tag
        self.image_start_tag = image_start_tag
        self.image_end_tag = image_end_tag

        self.num_image_tokens = num_image_tokens
        self.add_special_token = add_special_token
        self.sft_format = sft_format
        self.mask_prompt = mask_prompt
        self.ignore_id = ignore_id

        super().__init__(
            image_processor,
            tokenizer,
            image_tag,
            num_image_tokens,
            add_special_token,
            sft_format,
            mask_prompt,
            ignore_id,
            **kwargs,
        )

    #对话模板处理
    def new_chat_template(self): #生成新的对话模板
        conv = get_conv_template(self.sft_format)
        conv.set_system_message(self.system_prompt)
        return conv

    #将模板应用于多轮对话
    def apply_sft_template_for_multi_turn_prompts(
        self,
        conversations: List[Dict[str, str]],
        sft_format: str = "deepseek",
        system_prompt: str = "",
    ):
        """
        Applies the SFT template to conversation.

        An example of conversation:
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder> is Figure 1.\n<image_placeholder> is Figure 2.\nWhich image is brighter?",
                "images": [
                    "./multi-images/attribute_comparison_1.png",
                    "./multi-images/attribute_comparison_2.png"
                ]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        Args:
            conversations (List[Dict]): A conversation with a List of Dict[str, str] text.
            sft_format (str, optional): The format of the SFT template to use. Defaults to "deepseek".
            system_prompt (str, optional): The system prompt to use in the SFT template. Defaults to "".

        Returns:
            sft_prompt (str): The formatted text.
        """

        conv = get_conv_template(sft_format)
        conv.set_system_message(system_prompt)
        for message in conversations:
            conv.append_message(message["role"], message["content"].strip())#获取消息角色和内容，并去除多余空白字符
        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    @property 
    def image_token(self):
        return self.image_tag#返回图像占位符标记（默认 <image_placeholder>）

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)#返回图像占位符的标记 ID
        return image_id

    @property
    def image_start_id(self):
        image_start_id = self.tokenizer.vocab.get(self.image_start_tag)##获取图像起始标记 <begin_of_image> 对应的标记 ID
        return image_start_id

    @property
    def image_end_id(self):
        image_end_id = self.tokenizer.vocab.get(self.image_end_tag)
        return image_end_id

    @property
    def image_start_token(self):
        return self.image_start_tag#直接返回起始标记字符串 <begin_of_image>

    @property
    def image_end_token(self):
        return self.image_end_tag

    @property
    def pad_id(self):#返回填充标记（pad_id）的 ID，用于对输入序列进行对齐
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        return pad_id

     #图像标记处理：在输入文本序列中插入图像标记（起始标记、图像标记和结束标记），返回更新后的标记序列和图像标记数量
    def add_image_token(
        self,
        image_indices: List[int],# 图像插入位置的索引列表
        input_ids: torch.LongTensor,#输入文本的标记 ID（张量形式）
    ):
        """

        Args:
            image_indices (List[int]): [index_0, index_1, ..., index_j]
            input_ids (torch.LongTensor): [N]

        Returns:
            input_ids (torch.LongTensor): [N + image tokens]
            num_image_tokens (torch.IntTensor): [n_images]
        """

        input_slices = []

        start = 0
        for index in image_indices:
            if self.add_special_token:
                end = index + 1
            else:
                end = index

            # original text tokens
            input_slices.append(input_ids[start:end])#将输入文本从 start 到 end 的标记片段切分，并存入 input_slices

            # add boi, image tokens, eoi and set the mask as False
            input_slices.append(self.image_start_id * torch.ones((1), dtype=torch.long))## 添加图像的开始标记
            input_slices.append(
                self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
            )#添加图像标记，占用 `self.num_image_tokens` 个位置
            input_slices.append(self.image_end_id * torch.ones((1), dtype=torch.long))# 添加图像的结束标记
            start = index + 1#更新 start，指向下一个文本段的起始位置

        # the left part
        input_slices.append(input_ids[start:])#将切分后的剩余文本部分追加到 input_slices 中
        
        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)#将 input_slices 中的所有张量沿着第一个维度（dim=0）拼接起来，形成一个新的张量
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))#长度为len(image_indices),每个元素都是num_image_tokens的张量

        return input_ids, num_image_tokens

    #数据处理：将对话内容、图像和相应的标记进行编码，并创建一个包含必要信息的输出对象
    def process_one(
        self,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - target_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        assert (
            prompt is None or conversations is None
        ), "prompt and conversations cannot be used at the same time."

        if prompt is None:
            # apply sft format
            sft_format = self.apply_sft_template_for_multi_turn_prompts(
                conversations=conversations,
                sft_format=self.sft_format,
                system_prompt=self.system_prompt,
            )
        else:
            sft_format = prompt

        # tokenize
        input_ids = self.tokenizer.encode(sft_format)#将 sft_format（格式化后的对话内容）转换为标记 ID 列表，即模型所需的输入 ID
        input_ids = torch.LongTensor(input_ids)#将这个标记 ID 列表转换成 PyTorch 的长整型张量（LongTensor），以便后续处理。

        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == self.image_id#生成一个布尔张量，标记 input_ids 中等于图像标记 ID 的位置。
        image_indices = image_token_mask.nonzero()#返回一个包含所有图像标记位置的张量索引
        input_ids, num_image_tokens = self.add_image_token(
            image_indices=image_indices,
            input_ids=input_ids,
        )

        # load images
        images_outputs = self.image_processor(images, return_tensors="pt")

        prepare = VLChatProcessorOutput(
            sft_format=sft_format,
            input_ids=input_ids,
            pixel_values=images_outputs.pixel_values,
            num_image_tokens=num_image_tokens,
        )

        return prepare

    #调用处理器：高层接口，用于统一处理单条或批量输入
    def __call__(
        self,
        *,
        prompt: str = None,
        conversations: List[Dict[str, str]] = None,
        images: List[Image] = None,
        force_batchify: bool = True,
        **kwargs,
    ):
        """

        Args:
            prompt (str): the formatted prompt;
            conversations (List[Dict]): conversations with a list of messages;
            images (List[ImageType]): the list of images;
            force_batchify (bool): force batchify the inputs;
            **kwargs:

        Returns:
            outputs (BaseProcessorOutput): the output of the processor,
                - input_ids (torch.LongTensor): [N + image tokens]
                - images (torch.FloatTensor): [n_images, 3, H, W]
                - image_id (int): the id of the image token
                - num_image_tokens (List[int]): the number of image tokens
        """

        prepare = self.process_one(
            prompt=prompt, conversations=conversations, images=images
        )

        if force_batchify:
            prepare = self.batchify([prepare])

        return prepare

    #批量处理 将多个处理后的数据样本批量化，以便能够在多模态推理（同时处理文本和图像）时一起传递给模型进行处理
    def batchify(
        self, prepare_list: List[VLChatProcessorOutput]
    ) -> BatchedVLChatProcessorOutput:
        """
        Preprocesses the inputs for multimodal inference.

        Args:
            prepare_list (List[VLChatProcessorOutput]): A list of VLChatProcessorOutput.

        Returns:
            BatchedVLChatProcessorOutput: A dictionary of the inputs to use for multimodal inference.
        """

        batch_size = len(prepare_list)
        sft_format = []
        n_images = []
        seq_lens = []
        for prepare in prepare_list:#遍历 prepare_list 中的每个 VLChatProcessorOutput，将每个样本的图像数量 (n_images) 和序列长度 (seq_lens) 记录下来。
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))#是样本中包含图像的最大数量

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  # FIXME 创建一个形状为 (batch_size, input_token_max_len) 的张量，初始值为 pad_id，表示填充的标记ID。
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()#创建一个形状为 (batch_size, input_token_max_len) 的张量，初始化为 0。用于在训练时区分实际的标记和填充标记，1 表示实际的标记，0 表示填充
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()#用于存储图像的像素值（每个图像的大小是 default_shape）
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()#用于标记哪些位置是图像标记。
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, self.num_image_tokens)
        ).bool()#用于标记每张图像的嵌入位置

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            # left-padding
            batched_attention_mask[i, -seq_len:] = 1#将当前样本的实际标记位置在 batched_attention_mask 中标记为 1，倒数 seq_len 开始的位置插入，即右侧对齐
            batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)#将当前样本的 input_ids 填充到 batched_input_ids 中
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id#标记哪些位置是图像标记

            if n_image > 0:#如果样本包含图像，填充 batched_pixel_values 和 batched_images_emb_mask
                batched_pixel_values[i, :n_image] = prepare.pixel_values
                for j, n_image_tokens in enumerate(prepare.num_image_tokens):
                    batched_images_emb_mask[i, j, :n_image_tokens] = True

            sft_format.append(prepare.sft_format)

        batched_prepares = BatchedVLChatProcessorOutput(
            input_ids=batched_input_ids,
            attention_mask=batched_attention_mask,
            pixel_values=batched_pixel_values,
            images_seq_mask=batched_images_seq_mask,
            images_emb_mask=batched_images_emb_mask,
            sft_format=sft_format,
        )

        return batched_prepares
