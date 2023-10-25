# @Time    : 2023/3/25 18:36
# @Author  : tk
import copy
import numpy as np
from datasets import Audio
from transformers import PreTrainedTokenizer


class TokenIdsMaker:
    @classmethod
    def process(cls, data_args,
                tokenizer: PreTrainedTokenizer,
                config,
                max_seq_length,
                feature_extractor,
                examples,
                phoneme_language=None):
        max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
        min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
        sampling_rate = data_args.sampling_rate or feature_extractor.sampling_rate
        d = {}
        path,sentence = examples

        sample = Audio(sampling_rate=sampling_rate).decode_example({
            "path": path,"bytes": None
        })
        length = len(sample["array"])
        if length <= min_input_length and length > max_input_length:
            return None

        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"]
        )
        input_values = inputs["input_values"][0]
        d["shape"] = np.asarray(list(input_values.shape),dtype=np.int32)
        d["input_values"] = input_values.reshape(-1)

        # encode targets
        additional_kwargs = {}
        if phoneme_language is not None:
            additional_kwargs["phonemizer_lang"] = phoneme_language

        # process targets
        input_ids = tokenizer(sentence,**additional_kwargs).input_ids
        labels = input_ids[:max_seq_length] if max_seq_length > 0 else input_ids
        d["labels"] = np.asarray(labels,dtype=np.int32)
        return d



