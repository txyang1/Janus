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


@dataclass #数据类 
class VLChatProcessorOutput(DictOutput):#表示单个数据实例的处理结果。
    sft_format: str #用于生成对话的格式
    input_ids: torch.Tensor #文本输入的张量。
    pixel_values: torch.Tensor #图像的像素张量。
    num_image_tokens: torch.IntTensor #图像对应的标记数量

    def __len__(self):
        return len(self.input_ids)


@dataclass
class BatchedVLChatProcessorOutput(DictOutput):#表示批量数据的处理结果
    sft_format: List[str]
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    attention_mask: torch.Tensor
    images_seq_mask: torch.BoolTensor
    images_emb_mask: torch.BoolTensor

    def to(self, device, dtype=torch.bfloat16):
        self.input_ids = self.input_ids.to(device)
        self.attention_mask = self.attention_mask.to(device)
        self.images_seq_mask = self.images_seq_mask.to(device)
        self.images_emb_mask = self.images_emb_mask.to(device)
        self.pixel_values = self.pixel_values.to(device=device, dtype=dtype)
        return self

#主处理器类
class VLChatProcessor(ProcessorMixin): #静态属性：
    image_processor_class = "AutoImageProcessor" #指定默认图像处理类
    tokenizer_class = ("LlamaTokenizer", "LlamaTokenizerFast") #支持的分词器类型

    attributes = ["image_processor", "tokenizer"]

    system_prompt = (
        "You are a helpful language and vision assistant. "
        "You are able to understand the visual content that the user provides, "
        "and assist the user with a variety of tasks using natural language."
    )#默认的系统提示

    #构造方法：初始化主要组件（图像处理器和分词器）和相关配置。如果 image_tag 未在词汇表中，向分词器添加特殊标记
    def __init__(
        self,
        image_processor: VLMImageProcessor,
        tokenizer: LlamaTokenizerFast,
        image_tag: str = "<image_placeholder>",
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

    #方法功能：对话模板处理：
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
            conv.append_message(message["role"], message["content"].strip())
        sft_prompt = conv.get_prompt().strip()

        return sft_prompt

    @property #标记属性 通过属性访问特定标记和其对应的ID
    def image_token(self):
        return self.image_tag

    @property
    def image_id(self):
        image_id = self.tokenizer.vocab.get(self.image_tag)
        return image_id

    @property
    def image_start_id(self):
        image_start_id = self.tokenizer.vocab.get(self.image_start_tag)
        return image_start_id

    @property
    def image_end_id(self):
        image_end_id = self.tokenizer.vocab.get(self.image_end_tag)
        return image_end_id

    @property
    def image_start_token(self):
        return self.image_start_tag

    @property
    def image_end_token(self):
        return self.image_end_tag

    @property
    def pad_id(self):
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id

        return pad_id

     #图像标记处理：在输入中插入图像标记（起始标记、图像标记和结束标记）
    def add_image_token(
        self,
        image_indices: List[int],
        input_ids: torch.LongTensor,
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
            input_slices.append(input_ids[start:end])

            # add boi, image tokens, eoi and set the mask as False
            input_slices.append(self.image_start_id * torch.ones((1), dtype=torch.long))
            input_slices.append(
                self.image_id * torch.ones((self.num_image_tokens,), dtype=torch.long)
            )
            input_slices.append(self.image_end_id * torch.ones((1), dtype=torch.long))
            start = index + 1

        # the left part
        input_slices.append(input_ids[start:])

        # concat all slices
        input_ids = torch.cat(input_slices, dim=0)
        num_image_tokens = torch.IntTensor([self.num_image_tokens] * len(image_indices))

        return input_ids, num_image_tokens

    #数据处理方法：单条处理：接受文本和图像输入，生成处理结果
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
        input_ids = self.tokenizer.encode(sft_format)
        input_ids = torch.LongTensor(input_ids)

        # add image tokens to the input_ids
        image_token_mask: torch.BoolTensor = input_ids == self.image_id
        image_indices = image_token_mask.nonzero()
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

    #批量处理 将多个处理结果打包成批量数据，填充成相同长度。
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
        for prepare in prepare_list:
            n_images.append(len(prepare.num_image_tokens))
            seq_lens.append(len(prepare))

        input_token_max_len = max(seq_lens)
        max_n_images = max(1, max(n_images))

        batched_input_ids = torch.full(
            (batch_size, input_token_max_len), self.pad_id
        ).long()  # FIXME
        batched_attention_mask = torch.zeros((batch_size, input_token_max_len)).long()
        batched_pixel_values = torch.zeros(
            (batch_size, max_n_images, *self.image_processor.default_shape)
        ).float()
        batched_images_seq_mask = torch.zeros((batch_size, input_token_max_len)).bool()
        batched_images_emb_mask = torch.zeros(
            (batch_size, max_n_images, self.num_image_tokens)
        ).bool()

        for i, prepare in enumerate(prepare_list):
            input_ids = prepare.input_ids
            seq_len = len(prepare)
            n_image = len(prepare.num_image_tokens)
            # left-padding
            batched_attention_mask[i, -seq_len:] = 1
            batched_input_ids[i, -seq_len:] = torch.LongTensor(input_ids)
            batched_images_seq_mask[i, -seq_len:] = input_ids == self.image_id

            if n_image > 0:
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
