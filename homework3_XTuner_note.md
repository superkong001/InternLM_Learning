# 1 课程笔记

## 1.1 理论

### 指令微调：

<img width="405" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/f59cda06-ef89-4345-8161-e88ee401a035">

对话模版

<img width="662" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/ade3fbeb-906c-475a-b4c0-0e9d8c12b249">

<img width="773" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/279f622f-d56c-4654-a91e-435effa5109a">

<img width="779" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/6f45ac61-4dbc-4e23-a5dd-b66a0e455767">

### 增量预训练微调：

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

### 数据引擎

<img width="368" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/d6c60b91-b553-4ebb-8b23-8c7bb7c97be0">

<img width="691" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/7e6b4cac-338d-40f4-8973-c7f04b546d1b">

<img width="751" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/df12dedc-7684-442d-80e6-2ec42d737879">

<img width="774" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/14fe214b-1945-400e-9922-75967491c7d6">

<img width="696" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/aa130774-61a2-4172-96bf-3a3356177857">

QLoRA用deepspeed_zero2

<img width="656" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/7c189877-758f-4a6e-8523-d741e7598fc5">

## 1.2 操作
### 使用TMUX：(使用这个工具在终端SSH连接后，不会中断微调工作)

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

# 2 实操

## 安装
```Bash
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 2.0.1 的环境：
/root/share/install_conda_env_internlm_base.sh xtuner0.1.9

#!/bin/bash
# clone internlm-base conda env to user's conda env
# created by xj on 01.07.2024

echo "start cloning conda environment of internlm-base"

if [ -z "$1" ]; then
    echo "Error: No argument provided. Please provide a conda environment name."
    exit 1
fi


echo "uncompress pkgs.tar.gz file to /root/.conda/pkgs..."
sleep 3
tar --skip-old-files -xzvf /root/share/pkgs.tar.gz -C /root/.conda

echo "\n"
echo "uncompress internlm-base.tar.gz to /root/.conda/envs/$1..."
sleep 3
mkdir /root/.conda/envs/$1
tar --skip-old-files -xzvf /root/share/conda_envs/internlm-base.tar.gz -C /root/.conda/envs/$1
echo "Finised!"
echo "Now you can use your environment by typing:"
echo "conda activate $1"
```

```Bash
# begin：
root@intern-studio:~# bash
(base) root@intern-studio:~# conda create --name xtuner0.1.9 python=3.10 -y
(base) root@intern-studio:~# conda info -e
# 激活环境
conda activate xtuner0.1.9
# 进入家目录 （~的意思是 “当前用户的home路径”）
(xtuner0.1.9) root@intern-studio:~# cd ~
# 创建版本文件夹并进入，以跟随本教程
mkdir xtuner019 && cd xtuner019

# 拉取 0.1.9 的版本源码
# git clone -b v0.1.9  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
git clone -b v0.1.9 https://gitee.com/Internlm/xtuner

# 进入源码目录
cd xtuner

# 从源码安装 XTuner
pip install -e '.[all]'

# 创建一个微调 oasst1 数据集的工作路径，进入
mkdir ~/ft-oasst1 && cd ~/ft-oasst1
```

## 微调

### 准备配置文件

```Bash
# 列出所有内置配置
xtuner list-cfg
假如显示bash: xtuner: command not found的话可以考虑在终端输入 export PATH=$PATH:'/root/.local/bin'
```

<img width="844" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/088b3bd1-8dcc-490b-83be-1a0ea01fa083">

拷贝一个配置文件到当前目录：
```Bash
cd ~/ft-oasst1
xtuner copy-cfg internlm_chat_7b_qlora_oasst1_e3 .
```

| 模型名   | internlm_chat_7b |
| -------- | ---------------- |
| 使用算法 | qlora            |
| 数据集   | oasst1           |
| 把数据集跑几次    | 跑3次：e3 (epoch 3 )   |

### 模型下载

从 OpenXLab 下载模型到本地

```Bash
# 创建一个目录，放模型文件，防止散落一地
mkdir ~/ft-oasst1/internlm-chat-7b

# 装一下拉取模型文件要用的库
pip install modelscope

# 从 modelscope 下载下载模型文件
cd ~/ft-oasst1
apt install git git-lfs -y
git lfs install
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm-chat-7b.git -b v1.0.3
```

### 数据集下载

> https://huggingface.co/datasets/timdettmers/openassistant-guanaco/tree/main
> https://github.com/abachaa/Medication_QA_MedInfo2019

下载后放入新建的openassistant-guanaco目录

<img width="602" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/e6f46365-2a7c-417b-b4f4-0efff4d6a5a6">

目标格式：(.jsonL)

```Bash
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

写个python脚本，生成训练数据

```Bash
import json

def generate_conversations(replacement, n, filename):
    data = []
    for _ in range(n):
        conversation = {
            "conversation": [
                {
                    "input": "请介绍一下你自己",
                    "output": f"我是{replacement}的小助手,内在是上海AI实验室书生·浦语的7B大模型哦"
                }
            ]
        }
        data.append(conversation)

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    replacement = input("请输入需要替换的内容: ")
    n = int(input("请输入记录数 (n): "))
    filename = input("请输入输出文件名 (包括.json扩展名): ")

    generate_conversations(replacement, n, filename)
    print(f"已生成包含{n}条记录的文件: {filename}")

```

<img width="723" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/fe2227b5-8ddb-48b5-9f0f-ab802ac4cf22">

<img width="596" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/86af282b-cee3-4b26-bee4-27806d20c91e">

划分训练集和测试集....

### 修改配置文件

```Bash
# 改个文件名
mv internlm_chat_7b_qlora_oasst1_e3_copy.py internlm_chat_7b_qlora_mytrain_e3_copy.py

vim internlm_chat_7b_qlora_mytrain_e3_copy.py
```

减号代表要删除的行，加号代表要增加的行。

数据集的话：

```Bash
# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = './internlm-chat-7b'

# 修改训练数据集为本地路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = './openassistant-guanaco'

# 修改跑次数
- max_epochs = 3
+ max_epochs = 1
```

单个文件情况：
```Bash
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = '/root/ft-oasst1/internlm-chat-7b'

# 修改训练数据为 MedQA2019-structured-train.jsonl 路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = '/root/ft-oasst1/personal_assistant.json'

# 用于评估输出内容的问题（用于评估的问题尽量与数据集的question保持一致）
evaluation_freq = 90
SYSTEM = ''
evaluation_inputs = [
    '请介绍一下你自己', '请介绍一下你自己'
]

# 修改 train_dataset 对象
train_dataset = dict(
    type=process_hf_dataset,
-   dataset=dict(type=load_dataset, path=data_path),
+   dataset=dict(type=load_dataset, path='json', data_files=dict(train=data_path)),
    tokenizer=tokenizer,
    max_length=max_length,
-   dataset_map_fn=alpaca_map_fn,
+   dataset_map_fn=None,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    remove_unused_columns=True,
    shuffle_before_pack=True,
    pack_to_max_length=pack_to_max_length)

```

<img width="555" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/4d9bdff4-a23a-48cf-add6-f730c788b3dd">

<img width="502" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/b5e4a50c-c6c7-4cc1-a14f-e9dda3d9247f">


**常用超参**

| 参数名 | 解释 |
| ------------------- | ------------------------------------------------------ |
| **data_path**       | 数据路径或 HuggingFace 仓库名                          |
| max_length          | 单条数据最大 Token 数，超过则截断                      |
| pack_to_max_length  | 是否将多条短数据拼接到 max_length，提高 GPU 利用率     |
| accumulative_counts | 梯度累积，每多少次 backward 更新一次参数               |
| evaluation_inputs   | 训练过程中，会根据给定的问题进行推理，便于观测训练状态 |
| evaluation_freq     | Evaluation 的评测间隔 iter 数                          |
| ...... | ...... |

> 如果想把显卡的现存吃满，充分利用显卡资源，可以将 `max_length` 和 `batch_size` 这两个参数调大。

### 开始微调

使用TMUX：(使用这个工具在终端SSH连接后，不会中断微调工作)

```Bash
apt update -y
apt install tmux -y
创建并进入tumx环境
tmux new -s finetune 
退出Ctrl+B再按D，再进入
tmux attach -t finetune
```
写个python脚本，生成训练数据

训练：

```Bash
# 单卡
## 用刚才改好的config文件训练
xtuner train /root/ft-oasst1/internlm_chat_7b_qlora_mytrain_e3_copy.py --deepspeed deepspeed_zero2

# 多卡
NPROC_PER_NODE=${GPU_NUM} xtuner train internlm_chat_7b_qlora_mytrain_e3_copy.py --deepspeed deepspeed_zero2
# --deepspeed deepspeed_zero2, 开启 deepspeed 加速
```
<img width="572" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/8d500545-79bd-4da3-8dc4-bc064e60f581">

<img width="638" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/65e99b55-7555-41e6-a3e6-357e94e30058">

将得到的 PTH 模型转换为 HuggingFace 模型，即：生成 Adapter 文件夹
```Bash
mkdir hf
# 设置环境变量
export MKL_SERVICE_FORCE_INTEL=1

xtuner convert pth_to_hf internlm_chat_7b_qlora_mytrain_e3_copy.py ./work_dirs/internlm_chat_7b_qlora_mytrain_e3_copy/epoch_1.pth ./hf
```

<img width="845" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/03477e10-f7e4-4321-a2ca-1f971806f281">

### 部署与测试

将 HuggingFace adapter 合并到大语言模型：

```Bash
xtuner convert merge ./internlm-chat-7b ./hf ./merged --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

### 与合并后的模型对话：

```Bash
# 加载 Adapter 模型对话（Float 16）
# xtuner chat ./merged --prompt-template internlm_chat

# 4 bit 量化加载
xtuner chat ./merged --bits 4 --prompt-template internlm_chat
```

### Demo

```Bash
cd ~/ft-oasst1
cp ~/code/InternLM/cli_demo.py cli_demo.py
vim cli_demo.py

# 修改 cli_demo.py 中的模型路径
- model_name_or_path = "/root/model/Shanghai_AI_Laboratory/internlm-chat-7b"
+ model_name_or_path = "/root/ft-oasst1/merged"

# 运行 cli_demo.py 以目测微调效果
python cli_demo.py

cp ~/code/InternLM/web_demo.py web_demo.py
vim web_demo.py

# 修改
+ AutoModelForCausalLM.from_pretrained("/root/ft-oasst1/merged", trust_remote_code=True)
+ tokenizer = AutoTokenizer.from_pretrained("/root/ft-oasst1/merged", trust_remote_code=True)

pip install streamlit==1.24.0

# 创建code文件夹用于存放InternLM项目代码
mkdir /root/personal_assistant/code && cd /root/personal_assistant/code
git clone https://github.com/InternLM/InternLM.git

将 /root/code/InternLM/web_demo.py 中 29 行和 33 行的模型路径更换为Merge后存放参数的路径 /root/ft-oasst1/merged

streamlit run web_demo.py --server.address 127.0.0.1 --server.port 6006
```


**`xtuner chat`** **的启动参数**

| 启动参数              | 干哈滴                                                       |
| --------------------- | ------------------------------------------------------------ |
| **--prompt-template** | 指定对话模板                                                 |
| --system              | 指定SYSTEM文本                                               |
| --system-template     | 指定SYSTEM模板                                               |
| -**-bits**            | LLM位数                                                      |
| --bot-name            | bot名称                                                      |
| --with-plugins        | 指定要使用的插件                                             |
| **--no-streamer**     | 是否启用流式传输                                             |
| **--lagent**          | 是否使用lagent                                               |
| --command-stop-word   | 命令停止词                                                   |
| --answer-stop-word    | 回答停止词                                                   |
| --offload-folder      | 存放模型权重的文件夹（或者已经卸载模型权重的文件夹）         |
| --max-new-tokens      | 生成文本中允许的最大 `token` 数量                                |
| **--temperature**     | 温度值                                                       |
| --top-k               | 保留用于顶k筛选的最高概率词汇标记数                          |
| --top-p               | 如果设置为小于1的浮点数，仅保留概率相加高于 `top_p` 的最小一组最有可能的标记 |
| --seed                | 用于可重现文本生成的随机种子                                 |
