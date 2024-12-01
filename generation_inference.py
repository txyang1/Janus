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

import torch
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image

# specify the path to the model
model_path = "deepseek-ai/Janus-1.3B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

conversation = [
    {
        "role": "User",
        "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
    },
    {"role": "Assistant", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag


@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    temperature: float = 1,
    parallel_size: int = 16,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids#将完整的 input_ids（即文本 prompt 的 token 序列）复制到第 i 行
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id#中间部分（索引 1:-1）用特殊的填充值 pad_id 替换

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    for i in range(image_token_num_per_image):#每次生成一个图像的token,自回归
        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state# 通常是[batch_size, seq_len, hidden_dim]
        
        logits = mmgpt.gen_head(hidden_states[:, -1, :])#只提取最后一个 token 的隐藏状态，将隐藏状态映射到生成 token 的词汇表概率分布
        logit_cond = logits[0::2, :]#从 logits 张量中提取 偶数行（从第0行开始，步长为2），即条件生成的 logits 值，[batch_size, vocab_size]
        logit_uncond = logits[1::2, :]#从第一行开始，步长为2
        
        #Classifier-Free Guidance (CFG) 通过 CFG 方法增强文本描述对生成的影响
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        probs = torch.softmax(logits / temperature, dim=-1)#将 logits 转换为概率分布，控制生成时的随机性

        #采样下一个token
        next_token = torch.multinomial(probs, num_samples=1)#根据概率分布从词汇表中采样一个 token 作为输出
        generated_tokens[:, i] = next_token.squeeze(dim=-1)#假设 next_token 的形状为 [parallel_size, 1]，执行 squeeze(dim=-1) 后变为 [parallel_size]

        next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)#将 next_token 复制一份并沿着 dim=1 方向拼接，形状变为 [parallel_size, 2]，再将张量展平为一维数组。
        img_embeds = mmgpt.prepare_gen_img_embeds(next_token)#将 token 转换为图像嵌入
        inputs_embeds = img_embeds.unsqueeze(dim=1)#将图像嵌入用作下一时间步的输入，为 next_token 增加一个维度，使形状变为 [parallel_size, 1]


    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])#dec 是解码后的图像数据，形状为 [parallel_size, 8, 24, 24]。它表示每个图像的像素值，通常是浮动的连续值，表示图像的各个通道（例如 RGB）的强度
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)#转置后的形状为 [parallel_size, 24, 24, 8]

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)#适应常见的图像像素表示

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec#将解码后的图像数据 dec 赋值给 visual_img

    os.makedirs('generated_samples', exist_ok=True)#保存
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)


generate(
    vl_gpt,
    vl_chat_processor,
    prompt,
)
