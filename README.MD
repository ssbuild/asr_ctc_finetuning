## update information
   - [deep_training](https://github.com/ssbuild/deep_training)

```text
    10-24 initial asr ctc
```
   

## install
  - pip install -U -r requirements.txt
  - 如果无法安装， 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt



## weigtht select one is suitable for you
支持且不限于以下权重    
- [wav2vec2-base-100h](https://huggingface.co/facebook/wav2vec2-base-100h)
- [wav2vec2-base-960h](https://huggingface.co/facebook/wav2vec2-base-960h)
- [wav2vec2-large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)
- [wav2vec2-large-960h-lv60-self](https://huggingface.co/facebook/wav2vec2-large-960h-lv60-self)
- [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base)
- [wav2vec2-large](https://huggingface.co/facebook/wav2vec2-large)
- [wavlm-base-plus](https://huggingface.co/microsoft/wavlm-base-plus)
- [wavlm-base](https://huggingface.co/microsoft/wavlm-base)


## data sample
- open_data https://github.com/ssbuild/open_data
- librispeech_asr_dummy https://huggingface.co/datasets/patrickvonplaten/librispeech_asr_dummy
   
单条数据示例
```json
{"file": "../assets/librispeech_asr_dummy/1272-128104-0000.flac", "sentence": "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL"}
```



## infer
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
     python infer_finetuning.py



## training
```text
    # 制作数据
    cd scripts
    bash train_full.sh -m dataset 
    or
    bash train_lora.sh -m dataset 
    
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    # 全参数训练 
        bash train_full.sh -m train
        
    # lora adalora ia3 
        bash train_lora.sh -m train
```
   

## 训练参数
[训练参数](args.MD)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [chatmoss_finetuning](https://github.com/ssbuild/chatmoss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)

## 
    纯粹而干净的代码


## 参考

https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec#wav2vec-20

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/asr_ctc_finetuning&type=Date)](https://star-history.com/#ssbuild/asr_ctc_finetuning&Date)