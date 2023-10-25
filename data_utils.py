# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from typing import Union, Optional, List, Any
import warnings
import copy
import json
import random
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, TrainingArgumentsHF, \
    TrainingArgumentsCL, DataArguments, TrainingArgumentsAC
from aigc_zoo.model_zoo.asr_ctc.llm_model import PetlArguments,LoraConfig,PromptArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from transformers import PreTrainedTokenizer, HfArgumentParser, PretrainedConfig, Wav2Vec2Processor
from data_processer import TokenIdsMaker
from config import *
from module_setup import module_setup


module_setup()


def preprocess(text):
  return text

def postprocess(text):
  return text


class NN_DataHelper(DataHelper):
    index = 1
    padding: Union[bool, str] = "longest"
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __init__(self, *args, **kwargs):
        super(NN_DataHelper, self).__init__(*args, **kwargs)

    def load_tokenizer_and_config(self, *args, **kwargs):
        ret = super().load_tokenizer_and_config(*args, **kwargs)
        self._preprocess_tokenizer_config()
        self.load_feature_extractor()
        try:
            self.load_processer()
        except (OSError, KeyError):
            warnings.warn(
                "Loading a processor from a feature extractor config that does not"
                " include a `processor_class` attribute is deprecated and will be removed in v5. Please add the following "
                " attribute to your `preprocessor_config.json` file to suppress this warning: "
                " `'processor_class': 'Wav2Vec2Processor'`",
                FutureWarning,
            )
            self.processor = Wav2Vec2Processor.from_pretrained(training_args.output_dir)
        return ret

    def _preprocess_tokenizer_config(self):
       pass

    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: Any, mode: str):
        self.index += 1

        tokenizer: PreTrainedTokenizer
        config = self.config
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer
        feature_extractor = self.feature_extractor
        data_args = self.data_args
        examples = data


        d = TokenIdsMaker.process(data_args,
                tokenizer,
                config,
                max_seq_length,
                feature_extractor,
                examples)

        if not d:
            return None

        if self.index < 3:
            print(d)
        return d

    def _get_paragraph(self,lines):
        D = []
        for line_id, line in enumerate(lines):
            jd = json.loads(line)
            if not jd:
                continue
            D.append((jd["path"],jd["sentence"]))
        return D

    # 读取文件
    def on_get_corpus(self, files: List, mode: str):
        D = []
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()
            D.extend(self._get_paragraph(lines))
        return D

    def collate_fn(self, batch):
        batch = copy.copy(batch)
        model_input_name = "input_values"
        input_shape = [np.asarray(feature["shape"],dtype=np.int64) for feature in batch]
        input_features = [{model_input_name: np.asarray(feature[model_input_name],dtype=np.float32).reshape(input_shape[i])} for i,feature in enumerate(batch)]
        label_features = [{"input_ids": feature["labels"]} for feature in batch]

        o = self.processor.pad(
            input_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
            labels=label_features,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        o["labels"] = labels
        if "attention_mask" in o:
            o["attention_mask"] = o["attention_mask"].to(torch.long)
        return o

    def make_dataset_all(self):
        data_args = self.data_args
        # schema for arrow parquet
        schema = {
            "input_values": "float32_list",
            "shape": "int32_list",
            "labels": "int32_list",
        }


        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)



if __name__ == '__main__':

    if global_args["trainer_backend"] == "hf":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsHF, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    elif global_args["trainer_backend"] == "pl":
        parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments, PromptArguments))
        model_args, training_args, data_args, _, _ = parser.parse_dict(train_info_args)
    elif global_args["trainer_backend"] == "cl":
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsCL, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )
    else:
        parser = HfArgumentParser((ModelArguments, TrainingArgumentsAC, DataArguments, PetlArguments, PromptArguments),
                                  conflict_handler='resolve')
        model_args, training_args, data_args, lora_args, prompt_args = parser.parse_dict(train_info_args,
                                                                                         allow_extra_keys=True, )


    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(config_kwargs={"torch_dtype": torch.float16})
    



    # 缓存数据集
    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()


