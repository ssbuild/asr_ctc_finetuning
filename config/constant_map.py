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

MODELS_MAP = {
    'wavlm-base': {
        'model_type': 'wavlm',
        'model_name_or_path': '/data/nlp/pre_models/torch/wavlm/wavlm-base',
        'config_name': '/data/nlp/pre_models/torch/wavlm/wavlm-base-plus-v2/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/wavlm/wavlm-base-plus-v2',
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




