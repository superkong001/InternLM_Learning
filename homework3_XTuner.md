# 指令微调：

<img width="405" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/f59cda06-ef89-4345-8161-e88ee401a035">

对话模版

<img width="662" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/ade3fbeb-906c-475a-b4c0-0e9d8c12b249">

<img width="773" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/279f622f-d56c-4654-a91e-435effa5109a">

<img width="779" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/6f45ac61-4dbc-4e23-a5dd-b66a0e455767">

# 增量预训练微调：

<img width="722" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/58929c33-16ae-4a2a-bbc4-f9082d1fc2ea">

<img width="774" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/6b15eac4-7894-4b00-ac52-c85c2259b3f1">

LoRA (stable diffusion https://baijiahao.baidu.com/s?id=1765945330332903494&wfr=spider&for=pc)

<img width="683" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/51607873-7b31-4926-8175-56f784dc3dce">

XTuner

<img width="754" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/376ff3bf-1f42-46e0-8e70-42e12f10057b">

包括支撑Mistral、z

<img width="743" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/6378de28-b721-4950-a7b1-717ddc7a03d3">

<img width="744" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/e8d21a35-508c-4ee1-a199-39faa780d0d1">

生成Adapter(LoRA)

<img width="455" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/85ada56c-07bd-470b-80dd-2598074e35fc">

XTuner 还支持工具类模型的对话,更多详见HuggingFace Hub(xtuner/Llama-2-7b-qlora-moss-003-sft)

# 数据引擎

<img width="368" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/d6c60b91-b553-4ebb-8b23-8c7bb7c97be0">

<img width="691" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/7e6b4cac-338d-40f4-8973-c7f04b546d1b">

<img width="751" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/df12dedc-7684-442d-80e6-2ec42d737879">

<img width="774" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/14fe214b-1945-400e-9922-75967491c7d6">

<img width="696" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/aa130774-61a2-4172-96bf-3a3356177857">

QLoRA用deepspeed_zero2

<img width="656" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/7c189877-758f-4a6e-8523-d741e7598fc5">

# 使用TMUX：(使用这个工具在终端SSH连接后，不会中断微调工作)

apt update -y

apt install tmux -y

创建并进入tumx环境
tmux new -s finetune 

退出Ctrl+B再按D，再进入

tmux attach -t finetune

internlm_chat_7b_qlora_oasst1_e3 (模型名_方法qlora/lora_数据集_epoch3代表3轮，输出3个lora文件)

#单卡
##用刚才改好的config文件训练
xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2

#多卡
NPROC_PER_NODE=${GPU_NUM} xtuner train ./internlm_chat_7b_qlora_oasst1_e3_copy.py --deepspeed deepspeed_zero2

#--deepspeed deepspeed_zero2, 开启 deepspeed 加速

LoRA文件转成HuggingFace格式：

mkdir hf
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf ./internlm_chat_7b_qlora_oasst1_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_oasst1_e3_copy/epoch_1.pth ./hf

将 HuggingFace adapter 合并到大语言模型：max-shard-size切分的每个文件分块大小

xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB

与合并后的模型对话：

#加载 Adapter 模型对话（Float 16）
xtuner chat ./merged --prompt-template internlm_chat

#4 bit 量化加载
xtuner chat ./merged --bits 4 --prompt-template internlm_chat

--temperature	温度值，值0~1，越大回复越随机
--seed	用于可重现文本生成的随机种子，指定后可以保证每次随机种子一致

# 自定义微调





