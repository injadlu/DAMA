# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import io
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import random
import torch
import difflib
import transformers
import tokenizers
from torch.nn import Module
from functools import partial
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer

from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
import torch.nn.functional as F
from PIL import Image
import torch.distributed as dist
from transformers import Trainer
from torch.utils.data.sampler import SequentialSampler
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
class Colors:
    """ ANSI color codes """
    BLACK = "\033[0;30m"
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    BROWN = "\033[0;33m"
    BLUE = "\033[0;34m"
    PURPLE = "\033[0;35m"
    CYAN = "\033[0;36m"
    LIGHT_GRAY = "\033[0;37m"
    DARK_GRAY = "\033[1;30m"
    LIGHT_RED = "\033[1;31m"
    LIGHT_GREEN = "\033[1;32m"
    YELLOW = "\033[1;33m"
    LIGHT_BLUE = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_CYAN = "\033[1;36m"
    LIGHT_WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    FAINT = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    NEGATIVE = "\033[7m"
    CROSSED = "\033[9m"
    END = "\033[0m"
    # cancel SGR codes if we don't write to a terminal
    if not __import__("sys").stdout.isatty():
        for _ in dir():
            if isinstance(_, str) and _[0] != "_":
                locals()[_] = ""
    else:
        # set Windows console in VT mode
        if __import__("platform").system() == "Windows":
            kernel32 = __import__("ctypes").windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
            del kernel32

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_token_len: int = 0
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    data_source_names: str = 'unimm-chat'
    data_source_weights: str = '100'
    eval_data_source_names: Optional[str] = field(default=None)
    dpo_beta: float = 0.5
    dpo_token_weight: float = 3.0
    shuffle_data: bool = True
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    task: str = field(
        default='LM',
        metadata={
            'help': 'LM for language modeling. DPO for direct preference optimization'
        }
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    dpo_use_average: bool = False
    dpo_token_weighted: bool = False

    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

def expand_image_token(source, multimodal_cfg) -> Dict:
    is_multimodal = multimodal_cfg['is_multimodal']
    image_token_len = multimodal_cfg['image_token_len']
    if not is_multimodal or multimodal_cfg.get('keep_image_tag', False):
        return source

    for sentence in source:
        replace_token = DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
        if multimodal_cfg['use_im_start_end']:
            replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
        sentence["value"] = sentence["value"].replace(
            DEFAULT_IMAGE_TOKEN, replace_token)

    return source

def encode_multimodal_preference_sample(source, tokenizer, multimodal_cfg, preprocess_func=None):
    if isinstance(source['chosen'], list):
        win_conv = source['chosen']
        rej_conv = source['rejected']
    elif isinstance(source['chosen'], dict):
        win_conv = copy.deepcopy([source['question'], source["chosen"]])
        rej_conv = copy.deepcopy([source['question'], source["rejected"]])

    if 'image' in source:
        image = source['image']
        image = multimodal_cfg['image_processor'](image)
        win_conv = expand_image_token(win_conv, multimodal_cfg)
        rej_conv = expand_image_token(rej_conv, multimodal_cfg)

    if preprocess_func is None:
        rej_data_dict = preprocess([rej_conv], tokenizer)
        rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                             labels=rej_data_dict["labels"][0])

        win_data_dict = preprocess([win_conv], tokenizer)
        win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                             labels=win_data_dict["labels"][0])
    else:
        rej_data_dict = preprocess_func([rej_conv], tokenizer)
        win_data_dict = preprocess_func([win_conv], tokenizer)

        if 'context_ids' in rej_data_dict:
            rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                                labels=rej_data_dict["labels"][0],
                                image_bounds=rej_data_dict['image_bounds'][0],
                                context_ids=rej_data_dict['context_ids'][0],
                                position_ids=rej_data_dict['position_ids'][0]
                                )
            win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                                labels=win_data_dict["labels"][0],
                                image_bounds=win_data_dict['image_bounds'][0],
                                context_ids=win_data_dict['context_ids'][0],
                                position_ids=win_data_dict['position_ids'][0]
                                )
        else:
            rej_data_dict = dict(input_ids=rej_data_dict["input_ids"][0],
                                labels=rej_data_dict["labels"][0])
            win_data_dict = dict(input_ids=win_data_dict["input_ids"][0],
                                labels=win_data_dict["labels"][0])

    # image exist in the data
    if 'image' in source:
        rej_data_dict['image'] = win_data_dict['image'] = image
    elif multimodal_cfg['is_multimodal']:
        # image does not exist in the data, but the model is multimodal
        crop_size = multimodal_cfg['image_processor'].crop_size
        rej_data_dict['image'] = win_data_dict['image'] = torch.zeros(
            3, crop_size['height'], crop_size['width'])

    if 'ref_win_logp' in source:
        rej_data_dict['ref_rej_logp'] = source['ref_rej_logp']
        win_data_dict['ref_win_logp'] = source['ref_win_logp']
        rej_data_dict['ref_rej_avg_logp'] = source['ref_rej_avg_logp']
        win_data_dict['ref_win_avg_logp'] = source['ref_win_avg_logp']
        rej_data_dict['ref_rej_per_token_logp'] = source['ref_rej_per_token_logp']
        win_data_dict['ref_win_per_token_logp'] = source['ref_win_per_token_logp']
    
    if 'vlm_win_logits' in source:
        win_data_dict['vlm_win_logits'] = source['vlm_win_logits']
        rej_data_dict['vlm_rej_logits'] = source['vlm_rej_logits']
    
    return rej_data_dict, win_data_dict
class RLAIFVDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir: str, reference_model=None,
                 tokenizer=None, image_token_len=None, img_processor=None, use_im_start_end=True, is_llava15=False, image_folder=None):
        super().__init__()
        self.data = json.load(open(data_dir, "r"))
        self.image_folder = image_folder
        self.line_idx = list(range(len(self.data)))
        random.shuffle(self.line_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        sample = self.data[index]
        question = {'from': 'human', 'value': f"<image>\n{sample['question']}"}
        chosen = {'from': 'gpt', 'value': sample['chosen']}
        rejected = {'from': 'gpt', 'value': sample['rejected']}

        image_file = sample['image_path']
        image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')

        metainfo = {
            "origin_dataset": sample['origin_dataset'],
            "origin_split": sample['origin_split'],
            "origin_idx": sample['idx'],
            "image_id": sample['image_path'],
        }

        data_dict = {
            'image': image,
            "question": question,
            "chosen": chosen,
            "rejected": rejected,
            "idx": sample['idx'],
            "metainfo": metainfo
        }
        logps=json.loads(sample['logps'])

        if type(logps) == type([]):
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps
        else:
            (data_dict['ref_win_logp'], data_dict['ref_win_avg_logp'], data_dict['ref_win_per_token_logp'],
            data_dict['ref_rej_logp'], data_dict['ref_rej_avg_logp'], data_dict['ref_rej_per_token_logp']) = logps['logps']
        data_dict['vlm_rej_logits'] = sum(json.loads(sample['rejected_logits']))
        data_dict['vlm_win_logits'] = sum(json.loads(sample['chosen_logits']))
        return data_dict


class DPODataset(Dataset):
    def __init__(self,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_dir: str,
                 image_dir: str,
                 multimodal_cfg: dict,
                 reference_model = None):
        super(DPODataset, self).__init__()

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.list_data_dict = RLAIFVDataset(data_dir, reference_model, tokenizer,multimodal_cfg['image_token_len'], multimodal_cfg['image_processor'], multimodal_cfg['use_im_start_end'], is_llava15=True, image_folder=self.image_dir)
        self.multimodal_cfg = multimodal_cfg
        self.multimodal_cfg['keep_image_tag'] = True

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i):
        source: dict = self.list_data_dict[i]
        preprocess_func = partial(preprocess_v1, has_image=True)
        rej_data_dict, win_data_dict = encode_multimodal_preference_sample(
            source, self.tokenizer, self.multimodal_cfg, preprocess_func=preprocess_func)
        return rej_data_dict, win_data_dict

def SFT_collator_fn(instances, pad_token_id):
    input_ids, labels = tuple([instance[key] for instance in instances]
                              for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=pad_token_id)
    labels = torch.nn.utils.rnn.pad_sequence(labels,
                                             batch_first=True,
                                             padding_value=IGNORE_INDEX)
    batch = dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(pad_token_id),
    )

    images = [instance['image']
              for instance in instances if 'image' in instance]
    if len(images) > 0:
        # possibly multi-image for each sample
        if len(images[0].shape) == 4:
            batch['images'] = images
        elif all(x is not None and x.shape == images[0].shape for x in images):
            import numpy
            if isinstance(images[0], numpy.ndarray):
                images = [torch.from_numpy(x) for x in images]
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images
    else:
        batch['images'] = []

    # for minicpm
    if 'context_ids' in instances[0]:
        image_bounds, context_ids, position_ids = \
            tuple([instance[key] for instance in instances]
                  for key in ("image_bounds", "context_ids", "position_ids"))
        batch['image_bounds'] = image_bounds
        batch['context_ids'] = torch.nn.utils.rnn.pad_sequence(context_ids,
                                             batch_first=True,
                                             padding_value=0)
    return batch

def concate_pad(tensorA, tensorB, padding_value):
    out = torch.nn.utils.rnn.pad_sequence(
        list(tensorA) + list(tensorB),
        batch_first=True,
        padding_value=padding_value)
    return out

def preference_collator_fn(instances, pad_token_id):
    rej_instances, win_instances = list(zip(*instances))
    rej_batch = SFT_collator_fn(rej_instances, pad_token_id)
    win_batch = SFT_collator_fn(win_instances, pad_token_id)

    concatenated_input_ids = concate_pad(win_batch['input_ids'], rej_batch['input_ids'], pad_token_id)
    concatenated_labels = concate_pad(win_batch['labels'], rej_batch['labels'], -100)
    concatenated_attention_mask = concatenated_input_ids.ne(pad_token_id)

    batch = dict(
        concatenated_input_ids=concatenated_input_ids,
        concatenated_labels=concatenated_labels,
        concatenated_attention_mask=concatenated_attention_mask,
        win_input_ids=win_batch['input_ids'],
        rej_input_ids=rej_batch['input_ids'],
        win_labels=win_batch['labels'],
        rej_labels=rej_batch['labels'],
        win_attention_mask=win_batch['attention_mask'],
        rej_attention_mask=rej_batch['attention_mask'],
        images=win_batch['images'],
    )
    return batch

def complete_modification_spans(matches, length):
    i, j = 0, matches[0][0]
    out = []
    for idx in range(0, len(matches)):
        out.append((i, j))
        out.append(matches[idx])
        if idx + 1 < len(matches):
            i, j = matches[idx][1], matches[idx + 1][0]
        else:
            i, j = matches[idx][1], length
    return out

def join_by_space(seq):
    return ' '.join([str(x) for x in seq])

def colorize(raw_text, color):
    return f'{color}{raw_text}{Colors.END}'

def span_not_empty(span):
    return span[0] != span[1]

def generate_modification_mapping_impl(a_seq, b_seq, a_spans, b_spans, do_print=False):
    assert len(a_spans) == len(b_spans)
    mod_map = {}

    if do_print:
        print(a_spans)
        print(b_spans)

    for idx, (a_span, b_span) in enumerate(zip(a_spans, b_spans)):
        if idx % 2 == 1:
            continue
        a_text = join_by_space(a_seq[a_span[0]: a_span[1]])
        b_text = join_by_space(b_seq[b_span[0]: b_span[1]])
        if do_print:
            print(f'@{colorize(a_text, Colors.RED)}@ ==> @{colorize(b_text, Colors.GREEN)}@')

        if span_not_empty(a_span) and span_not_empty(b_span):
            mod_map[a_span] = b_span

    return mod_map


def get_match_info(a_seq, b_seq, min_match_size=1):
    sm = difflib.SequenceMatcher(None, a_seq, b_seq)

    mb = sm.get_matching_blocks()

    mb = [m for m in mb[:-1] if m[2] >= min_match_size] + [mb[-1]]

    a_matches = [(x[0], x[0] + x[2]) for x in mb]
    b_matches = [(x[1], x[1] + x[2]) for x in mb]
    return a_matches, b_matches

def generate_modification_mapping(a_seq, b_seq, min_match_size=3, do_print=False):
    a_matches, b_matches = get_match_info(a_seq, b_seq, min_match_size=min_match_size)

    a_spans = complete_modification_spans(a_matches, len(a_seq))
    b_spans = complete_modification_spans(b_matches, len(b_seq))
    return generate_modification_mapping_impl(a_seq, b_seq, a_spans, b_spans, do_print=do_print)


def spans2ids(spans):
    ids = []
    for span in spans:
        ids += list(range(span[0], span[1]))
    return ids

def get_diff_ids(a_seq, b_seq, min_match_size=3):
    mod_map = generate_modification_mapping(a_seq, b_seq, min_match_size=min_match_size)
    a_modification_spans = list(mod_map.keys())
    b_modification_spans = list(mod_map.values())

    a_ids = sorted(set(spans2ids(a_modification_spans)))
    b_ids = sorted(set(spans2ids(b_modification_spans)))
    return a_ids, b_ids


@dataclass
class DataCollatorForDPODataset(object):
    tokenizer: transformers.PreTrainedTokenizer
    beta: float
    mod_token_weight: float

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = preference_collator_fn(instances, self.tokenizer.pad_token_id)

        rej_instances, win_instances = list(zip(*instances))

        batch['beta'] = self.beta

        batch['ref_win_logp'] = torch.as_tensor(
            [x['ref_win_logp'] for x in win_instances])
        batch['ref_rej_logp'] = torch.as_tensor(
            [x['ref_rej_logp'] for x in rej_instances])
        batch['ref_win_avg_logp'] = torch.as_tensor(
            [x['ref_win_avg_logp'] for x in win_instances])
        batch['ref_rej_avg_logp'] = torch.as_tensor(
            [x['ref_rej_avg_logp'] for x in rej_instances])
        batch['vlm_win_logits'] = torch.as_tensor(
            [x['vlm_win_logits'] for x in win_instances])
        batch['vlm_rej_logits'] = torch.as_tensor(
            [x['vlm_rej_logits'] for x in rej_instances])
        ref_win_per_token_logp = [torch.as_tensor(
            x['ref_win_per_token_logp']) for x in win_instances]
        ref_rej_per_token_logp = [torch.as_tensor(
            x['ref_rej_per_token_logp']) for x in rej_instances]

        batch['ref_win_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(
            ref_win_per_token_logp, batch_first=True, padding_value=0)
        batch['ref_rej_per_token_logp'] = torch.nn.utils.rnn.pad_sequence(
            ref_rej_per_token_logp, batch_first=True, padding_value=0)

        win_input_ids = batch['win_input_ids']
        rej_input_ids = batch['rej_input_ids']
        win_labels = batch['win_labels']
        rej_labels = batch['rej_labels']
        assert batch['ref_win_per_token_logp'].size(1) >= win_input_ids.size(
            1) - 1, f"{batch['ref_win_per_token_logp'].size(1)} >= {win_input_ids.size(1) - 1}"
        assert batch['ref_rej_per_token_logp'].size(1) >= rej_input_ids.size(
            1) - 1, f"{batch['ref_rej_per_token_logp'].size(1)} >= {rej_input_ids.size(1) - 1}"

        # length of logp is one-token shorter since the last token's output is not used
        batch['ref_win_per_token_logp'] = batch['ref_win_per_token_logp'][:,
                                                                          :win_input_ids.size(1) - 1]
        batch['ref_rej_per_token_logp'] = batch['ref_rej_per_token_logp'][:,
                                                                          :rej_input_ids.size(1) - 1]

        win_token_weight = torch.ones_like(batch['ref_win_per_token_logp'])
        rej_token_weight = torch.ones_like(batch['ref_rej_per_token_logp'])

        for idx, (w, r, wl, rl, wlogp, rlogp) in enumerate(zip(win_input_ids, rej_input_ids, win_labels, rej_labels, ref_win_per_token_logp, ref_rej_per_token_logp)):
            valid_w = w[1:]
            valid_r = r[1:]
            min_match_size = 3
            r_mod, w_mod = get_diff_ids(
                valid_r.tolist(), valid_w.tolist(), min_match_size=min_match_size)
            r_mod_tokens = valid_r[r_mod]
            w_mod_tokens = valid_w[w_mod]
            win_token_weight[idx][w_mod] = self.mod_token_weight
            rej_token_weight[idx][r_mod] = self.mod_token_weight

        batch['win_token_weight'] = win_token_weight
        batch['rej_token_weight'] = rej_token_weight
        batch['concatenated_token_weight'] = concate_pad(
            win_token_weight, rej_token_weight, 0)

        for ins in win_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        for ins in rej_instances:
            assert len(ins['input_ids']) == len(ins['labels'])
        if torch.any(torch.isnan(batch['win_token_weight'])):
            print(f'win_token_weight fail', flush=True)
            exit()
        if torch.any(torch.isnan(batch['rej_token_weight'])):
            print(f'rej_token_weight fail', flush=True)
            exit()
        return batch

def make_dpo_data_module(tokenizer, data_args,reference_model):
    train_dataset = DPODataset(tokenizer=tokenizer,
                               data_dir=data_args.data_path,
                               image_dir=data_args.image_folder,
                               multimodal_cfg=dict(
                                   is_multimodal=data_args.is_multimodal,
                                   image_token_len=data_args.image_token_len,
                                   image_aspect_ratio=data_args.image_aspect_ratio,
                                   use_im_start_end=getattr(
                                       data_args, 'mm_use_im_start_end', False),
                                   image_processor=getattr(
                                       data_args, 'image_processor', None),
                                   data_source_names=getattr(
                                       data_args, 'data_source_names'),
                                   data_source_weights=getattr(data_args, 'data_source_weights'),
                                   shuffle_data=data_args.shuffle_data
                                   ),
                               reference_model=reference_model)
    print(f'Train data size is {len(train_dataset)}', flush=True)
    data_collator = DataCollatorForDPODataset(
        tokenizer=tokenizer, beta=data_args.dpo_beta, mod_token_weight=data_args.dpo_token_weight)

    if data_args.eval_data_source_names is not None:
        eval_datasets = {}
        for name in data_args.eval_data_source_names:
            eval_dataset = DPODataset(tokenizer=tokenizer,
                                      data_dir=data_args.data_path,
                                      image_dir=data_args.image_folder,
                                      multimodal_cfg=dict(
                                          is_multimodal=data_args.is_multimodal,
                                          image_token_len=data_args.image_token_len,
                                          image_aspect_ratio=data_args.image_aspect_ratio,
                                          use_im_start_end=getattr(
                                              data_args, 'mm_use_im_start_end', False),
                                          image_processor=getattr(
                                              data_args, 'image_processor', None),
                                          data_source_names=[name],
                                          data_source_weights=[1],
                                           shuffle_data=False
                                          ),
                                      reference_model=reference_model)
            eval_datasets[name] = eval_dataset
    else:
        eval_datasets = None

    return dict(train_dataset=train_dataset,
                eval_dataset=eval_datasets,
                data_collator=data_collator)

def forward_DPO(model, input_ids, labels, attention_mask, images, **kwargs):
    token_weighted = kwargs.pop('token_weighted', False)
    dpo_use_average = kwargs.pop('dpo_use_average', False)
    is_minicpm = kwargs.pop('is_minicpm', False)

    output = model(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        images=images,
        **kwargs
    )
    impl = get_batch_logps
    if token_weighted:
        token_log_prob = impl(
            output.logits, labels, return_per_token_logp=True)
        return token_log_prob
    else:
        log_prob, average_log_prob = impl(
            output.logits, labels, return_per_token_logp=False)
        if dpo_use_average:
            return average_log_prob
        return log_prob

def all_gather_if_needed(values: torch.Tensor, rank: int, world_size: int) -> torch.Tensor:
    """Gather and stack/cat values from all processes, if there are multiple processes."""
    if world_size == 1:
        return values

    all_values = [torch.empty_like(values).to(rank) for _ in range(world_size)]
    dist.all_gather(all_values, values)
    cat_function = torch.cat if values.dim() > 0 else torch.stack
    return cat_function(all_values, dim=0)

def get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, return_per_token_logp=False, return_all=False, tokenizer=None) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape, f'logits.shape[:-1]={logits.shape[:-1]}, labels.shape={labels.shape}'

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2,
                                   index=labels.unsqueeze(2)).squeeze(2)

    log_prob = (per_token_logps * loss_mask).sum(-1)
    average_log_prob = log_prob / loss_mask.sum(-1)

    # print("==>", labels)

    # print(per_token_logps.shape, labels.shape)
    if return_per_token_logp:
        return per_token_logps

    if return_all:
        return per_token_logps, log_prob, average_log_prob

    return log_prob, average_log_prob


def dpo_loss(policy_chosen_logps: torch.FloatTensor,
             policy_rejected_logps: torch.FloatTensor,
             reference_chosen_logps: torch.FloatTensor,
             reference_rejected_logps: torch.FloatTensor,
             vlm_chosen_logits: torch.FloatTensor,
             vlm_rejected_logits: torch.FloatTensor, 
             beta: float, rank, world_size, 
             gap_mean, gap_std, 
             reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0
    logits = pi_logratios - ref_logratios
    #####################################################
    logits_diff = (vlm_chosen_logits - vlm_rejected_logits)
    logits_diff = F.sigmoid(logits_diff)
    logits_mean = torch.tensor(0.0482190464911681, device='cuda')
    logits_mean = F.sigmoid(logits_mean)
    this_logits = logits_diff / logits_mean
    this_logits = this_logits.to(torch.float32)
    A_gap = (policy_chosen_logps - policy_rejected_logps - reference_chosen_logps + reference_rejected_logps)
    A = all_gather_if_needed(A_gap.detach(), rank, world_size)
    A_sigmoid = F.sigmoid(A)
    mean = gap_mean
    weight_sample = torch.exp(-0.5 * ((A - mean)).pow(2))
    #########################################################
    # no data
    sample_num = int(weight_sample.numel() * (1 - 0.2))
    if (weight_sample.sum() == 0.) or (torch.inf in weight_sample):
        print('INF in weight sample')
        sample_index = random.choices(range(len(weight_sample)), k=sample_num)
    else:
        sample_index = torch.multinomial(weight_sample, sample_num, replacement=False)
    one_hot_like = torch.zeros_like(weight_sample)
    one_hot_like[sample_index] = 1
    global_mask = one_hot_like.detach()
    global_mask = global_mask.to(torch.float32)
    ###########################################################
    # 2. compute the beta
    mean_sigmoid = F.sigmoid(mean)
    A_sigmoid_average = torch.mean(A_sigmoid[sample_index])
    this_ratio = A_sigmoid_average / mean_sigmoid
    this_ratio = this_logits * this_ratio

    beta_used = beta * this_ratio
    beta_clamped = (beta_used <= 1e-3).sum()
    beta_used = beta_used.clamp(min=1e-3)
    #####################################################
    losses = -F.logsigmoid(beta_used * logits)
    chosen_rewards = beta * (policy_chosen_logps -
                             reference_chosen_logps).detach()
    rejected_rewards = beta * \
        (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards, beta_used, global_mask, beta_clamped, A_sigmoid_average, this_ratio, this_logits

def compute_weighted_logp(per_token_logp, labels, token_weight, use_average):
    loss_mask = (labels[:, 1:].clone() != -100)
    weighted_mask = token_weight * loss_mask
    logp = (per_token_logp * weighted_mask).sum(-1)

    average_logp = logp / weighted_mask.sum(-1)
    if use_average:
        return average_logp
    return logp


class LLaVA15DPOTrainer(Trainer):
    def __init__(self, model, tokenizer, args, **data_module):
        super().__init__(model=model, tokenizer=tokenizer, args=args, **data_module)
        self.gap_mean = torch.zeros(1, device='cuda')
        self.gap_std = torch.zeros(1, device='cuda')
        self.loss_mean = torch.zeros(1, device='cuda')
        self.loss_std = torch.zeros(1, device='cuda')
        self.rank = self.args.local_rank
        self.world_size = self.args.world_size

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        return SequentialSampler(self.train_dataset)

    def compute_loss(self, model: Module, inputs: dict, return_outputs=False):
        if self.args.past_index >= 0:
            raise NotImplementedError

        def gather_and_do_mean(x):
            return self._nested_gather(x.mean()).mean().item()

        data_dict = inputs
        policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, vlm_win_logits, vlm_rej_logits, beta = self.get_beta_and_logps(
            data_dict, model, self.args, is_llava15=True)

        losses, chosen_rewards, rejected_rewards, beta_used, global_mask, beta_clamped, A_sigmoid_average, this_ratio, logits_diff = dpo_loss(policy_win_logp,
                                                                        policy_rej_logp,
                                                                        ref_win_logp,
                                                                        ref_rej_logp,
                                                                        vlm_win_logits,
                                                                        vlm_rej_logits,
                                                                        beta=beta, 
                                                                        rank=self.rank, 
                                                                        world_size=self.world_size, 
                                                                        gap_mean=self.gap_mean, 
                                                                        gap_std=self.gap_std)
        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        if global_mask is not None:
            B = int(global_mask.size(0) / self.world_size)
            local_mask = (global_mask[self.rank * B: (self.rank + 1) * B]).float()
            masked_losses = losses * local_mask
            effective_loss_count = local_mask.sum()
            effective_loss_count = effective_loss_count if effective_loss_count > 0 else 1
            loss = masked_losses.sum() / effective_loss_count
        else:
            loss = losses.mean()
        t = 'train' if model.training else 'test'
        metrics = {}
        metrics = self.collect_preference_metrics(metrics, t, chosen_rewards, rejected_rewards,
                                             policy_rej_logp, policy_win_logp,
                                             ref_rej_logp, ref_win_logp, vlm_win_logits, vlm_rej_logits, reward_accuracies, beta, beta_used, global_mask, beta_clamped, A_sigmoid_average, this_ratio, logits_diff,
                                             gather_and_do_mean)
        self.log(metrics)
        return loss

    def update_and_sync_tensor_mean(self, gap_local, loss_local, gamma=0.9):
        with torch.no_grad():
            batch_gap_mean = gap_local.mean()
            batch_gap_std = gap_local.std()
            batch_loss_mean = loss_local.mean()
            batch_loss_std = loss_local.std()
            self.gap_mean.mul_(gamma).add_(batch_gap_mean, alpha=1-gamma)
            self.gap_std.mul_(gamma).add_(batch_gap_std, alpha=1-gamma)
            self.loss_mean.mul_(gamma).add_(batch_loss_mean, alpha=1-gamma)
            self.loss_std.mul_(gamma).add_(batch_loss_std, alpha=1-gamma)
            if self.world_size > 1:
                dist.all_reduce(self.gap_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.gap_std, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.loss_mean, op=dist.ReduceOp.SUM)
                dist.all_reduce(self.loss_std, op=dist.ReduceOp.SUM)
                self.gap_mean /= self.world_size
                self.gap_std /= self.world_size
                self.loss_mean /= self.world_size
                self.loss_std /= self.world_size
    def collect_preference_metrics(self, metrics, task,
                               chosen_rewards, rejected_rewards,
                               policy_rej_logp, policy_win_logp,
                               ref_rej_logp, ref_win_logp, vlm_win_logits, vlm_rej_logits, reward_accuracies, beta, beta_used, global_mask, beta_clamped, A_sigmoid_average, this_ratio, logits_diff, 
                               preprocess_func,
                               ):
        t = task
        metrics = {}
        gap_local = (policy_win_logp - policy_rej_logp - ref_win_logp + ref_rej_logp).detach()
        gap_local = all_gather_if_needed(gap_local, self.rank, self.world_size)
        loss_local = -F.logsigmoid(beta * gap_local)
        self.update_and_sync_tensor_mean(gap_local, loss_local)
        metrics[f'record_{t}/gap_mean'] = self.gap_mean.cpu().numpy().tolist()[0]
        metrics[f'record_{t}/gap_std'] = self.gap_std.cpu().numpy().tolist()[0]
        metrics[f'record_{t}/loss_mean'] = self.loss_mean.cpu().numpy().tolist()[0]
        metrics[f'record_{t}/loss_std'] = self.loss_std.cpu().numpy().tolist()[0]
        metrics[f'record_{t}/beta_used'] = beta_used
        metrics[f'record_{t}/beta_clamped'] = beta_clamped.cpu().numpy().tolist()
        metrics[f'record_{t}/mask_count'] = global_mask.sum().cpu().numpy().tolist()
        metrics[f'record_{t}/A_sigmoid'] = A_sigmoid_average.cpu().numpy().tolist()
        metrics[f'record_{t}/this_ratio'] = this_ratio.cpu().numpy().tolist()
        metrics[f'rewards_{t}/chosen'] = preprocess_func(chosen_rewards)
        metrics[f'rewards_{t}/rejected'] = preprocess_func(rejected_rewards)
        metrics[f'logps_{t}/rejected'] = preprocess_func(policy_rej_logp)
        metrics[f'logps_{t}/chosen'] = preprocess_func(policy_win_logp)
        metrics[f'logps_{t}/ref_rejected'] = preprocess_func(ref_rej_logp)
        metrics[f'logps_{t}/ref_chosen'] = preprocess_func(ref_win_logp)
        metrics[f'logps_{t}/vlm_rejected'] = preprocess_func(vlm_rej_logits)
        metrics[f'logps_{t}/vlm_chosen'] = preprocess_func(vlm_win_logits)
        metrics[f'logps_{t}/vlm_diff_sigmoid'] = logits_diff.cpu().numpy().tolist()
        metrics[f'rewards_{t}/accuracies'] = preprocess_func(reward_accuracies)
        metrics[f'rewards_{t}/margins'] = metrics[f'rewards_{t}/chosen'] - \
            metrics[f'rewards_{t}/rejected']
        return metrics
        
    
    def get_beta_and_logps(self, data_dict, model, args, is_minicpm=False, is_llava15=False):
        win_input_ids = data_dict.pop('win_input_ids')
        rej_input_ids = data_dict.pop('rej_input_ids')

        win_labels = data_dict.pop('win_labels')
        rej_labels = data_dict.pop('rej_labels')

        win_attention_mask = data_dict.pop('win_attention_mask')
        rej_attention_mask = data_dict.pop('rej_attention_mask')

        ref_win_avg_logp = data_dict.pop('ref_win_avg_logp')
        ref_rej_avg_logp = data_dict.pop('ref_rej_avg_logp')
        ref_win_logp = data_dict.pop('ref_win_logp')
        ref_rej_logp = data_dict.pop('ref_rej_logp')
        ref_win_per_token_logp = data_dict.pop('ref_win_per_token_logp')
        ref_rej_per_token_logp = data_dict.pop('ref_rej_per_token_logp')

        vlm_win_logits = data_dict.pop('vlm_win_logits')
        vlm_rej_logits = data_dict.pop('vlm_rej_logits')
        if args.dpo_use_average:
            ref_win_logp = ref_win_avg_logp
            ref_rej_logp = ref_rej_avg_logp

        beta = data_dict.pop('beta')
        if args.task == 'DPO':
            images = data_dict.pop('images')
            if is_minicpm:
                # print(data_dict.keys())
                data_dict.pop('win_context_ids')
                data_dict.pop('rej_context_ids')
                concatenated_images = images
            else:
                concatenated_images = torch.cat([images, images], dim=0)
        elif args.task == 'KTO':
            win_images = data_dict.pop('win_images')
            rej_images = data_dict.pop('rej_images')
            concatenated_images = torch.cat([win_images, rej_images], dim=0)

        concatenated_input_ids = data_dict.pop('concatenated_input_ids')
        concatenated_labels = data_dict.pop('concatenated_labels')
        concatenated_attention_mask = data_dict.pop('concatenated_attention_mask')
        concatenated_attention_mask = None

        win_token_weight = data_dict.pop('win_token_weight')
        rej_token_weight = data_dict.pop('rej_token_weight')
        concatenated_token_weight = data_dict.pop('concatenated_token_weight')
        if is_llava15:
            (
                _,
                _,
                _,
                _,
                concatenated_inputs_embeds,
                concatenated_labels
            ) = model.prepare_inputs_labels_for_multimodal(
                input_ids=concatenated_input_ids,
                position_ids=None,
                attention_mask=None,
                past_key_values=None,
                labels=concatenated_labels,
                images=concatenated_images,
            )
            output = model.forward(
                inputs_embeds=concatenated_inputs_embeds,
                labels=None,
                **data_dict,
            )
            log_prob, average_log_prob = get_batch_logps(
                output.logits, concatenated_labels, return_per_token_logp=False)

            if args.dpo_use_average:
                concatenated_logp = average_log_prob
            else:
                concatenated_logp =log_prob
        else:
            concatenated_logp = forward_DPO(model,
                                            concatenated_input_ids,
                                            concatenated_labels,
                                            concatenated_attention_mask,
                                            concatenated_images,
                                            token_weighted=args.dpo_token_weighted,
                                            dpo_use_average=args.dpo_use_average,
                                            is_minicpm=is_minicpm,
                                            **data_dict)
        win_size = win_input_ids.shape[0]
        rej_size = rej_input_ids.shape[0]
        assert win_size == rej_size

        if args.dpo_token_weighted:
            if is_llava15:
                raise NotImplementedError
            ref_win_logp = compute_weighted_logp(
                ref_win_per_token_logp, win_labels, win_token_weight, args.dpo_use_average)
            ref_rej_logp = compute_weighted_logp(
                ref_rej_per_token_logp, rej_labels, rej_token_weight, args.dpo_use_average)
            concatenated_logp = compute_weighted_logp(
                concatenated_logp, concatenated_labels, concatenated_token_weight, args.dpo_use_average)

            if torch.any(torch.isnan(ref_win_logp)):
                print(f'ref_win_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(ref_rej_logp)):
                print(f'ref_rej_logp fail', flush=True)
                exit()
            if torch.any(torch.isnan(concatenated_logp)):
                print(f'concatenated_logp fail', flush=True)
                exit()

        policy_win_logp, policy_rej_logp = concatenated_logp.split(
            [win_size, rej_size])
        return policy_win_logp, policy_rej_logp, ref_win_logp, ref_rej_logp, vlm_win_logits, vlm_rej_logits, beta


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                # attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )

    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = lambda x: vision_tower.image_processor(x)['pixel_values'][0]
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_dpo_data_module(tokenizer, data_args=data_args, reference_model=copy.deepcopy(model).cuda())

    trainer = LLaVA15DPOTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
        print('Resume from Checkpoints.')
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
