# coding=utf8
# @Time    : 2023/10/26 0:07
# @Author  : tk
# @FileName: make_data
import json
import os.path
from shutil import copyfile
import datasets

ds = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy","clean")

with open('train.json',mode='w',encoding='utf-8',newline='\n') as f:
    for d in ds["validation"]:
        print(d)
        file = d["file"]
        text = d["text"]

        copyfile(file,os.path.basename(file))

        f.write(json.dumps({
            "path": "../assets/librispeech_asr_dummy/" + os.path.basename(file),
            "sentence": text
        },ensure_ascii=False) + '\n')