# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "MODELS_MAP"
]

# - [wav2vec2-base-100h](https://huggingface.co/facebook/wav2vec2-base-100h)
# - [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
# - [wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)
# - [wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)
# - [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
# - [wav2vec2-large](https://huggingface.co/facebook/wav2vec2-large)



MODELS_MAP = {
    'wav2vec2-large': {
        'model_type': 'wav2vec2',
        'model_name_or_path': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-large',
        'config_name': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-large/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-large',
    },
    
    'wav2vec2-base': {
        'model_type': 'wav2vec2',
        'model_name_or_path': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-base',
        'config_name': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-base/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-base',
    },
    
    'wav2vec2-large-960h': {
        'model_type': 'wav2vec2',
        'model_name_or_path': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-large-960h',
        'config_name': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-large-960h/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/wav2vec2/wav2vec2-large-960h',
    },
    
    'wavlm-base': {
        'model_type': 'wavlm',
        'model_name_or_path': '/data/nlp/pre_models/torch/wavlm/wavlm-base',
        'config_name': '/data/nlp/pre_models/torch/wavlm/wavlm-base/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/wavlm/wavlm-base',
    },
    'wavlm-base-plus': {
        'model_type': 'wavlm',
        'model_name_or_path': '/data/nlp/pre_models/torch/wavlm/wavlm-base-plus',
        'config_name': '/data/nlp/pre_models/torch/wavlm/wavlm-base-plus/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/wavlm/wavlm-base-plus',
    },



}


# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING




