# QLora微调

## 环境部署

```Bash
conda create --name solomon_chart python=3.10 -y
conda info -e
conda activate solomon_chart

# 安装xtuner（v0.1.15）
mkdir ~/xtuner && cd ~/xtuner
git clone https://github.com/InternLM/xtuner.git
cd xtuner
# tips: xtuner的requests==2.31.0与openxlab的版本冲突、transformers>=4.34.0,!=4.34.1,!=4.35.0,!=4.35.1,!=4.35.2
pip install -e '.[all]'
# 列出所有内置配置
xtuner list-cfg

# 模型下载
# 创建一个目录，放模型文件
cd ~/model
```

download_model.py

```Bash
import torch
from modelscope import snapshot_download, AutoModel, AutoTokenizer
import os
model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm2-chat-7b', cache_dir='/root/model')
```

```Bash
# 创建一个微调 solomon 数据集的工作路径，进入
mkdir ~/solomon && cd ~/solomon

ln -s /root/model/Shanghai_AI_Laboratory/internlm2-chat-7b ~/solomon/
```

## 准备Qlora数据集
```Bash
mkdir ~/solomon/data/dataset && cd ~/solomon/data/train_data

编写excel_to_json.py实现将excel文件转换为单轮对话的json格式
编写process_txt_to_json.py实现将目录下的txt转换为单轮对话的json格式(txt文档是亚里士多德写的哲学著作:))
excel格式：第一列是system内容,第二列是input内容,第三列是output内容
Aristotle.xlsx
Socrates.xlsx
Plato.xlsx

python excel_to_json.py Aristotle.xlsx
python 编写process_txt_to_json.py data
```

```Bash
# example
# 单轮对话数据格式
[{
    "conversation":[
        {
            "system": "请你扮演哲学家亚里士多德，请以他的哲学思想和口吻回答问题。",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "请你扮演哲学家苏格拉底，请以他的哲学思想和口吻回答问题。",
            "input": "xxx",
            "output": "xxx"
        }
    ]
},
{
    "conversation":[
        {
            "system": "请你扮演哲学家柏拉图，请以他的哲学思想和口吻回答问题。",
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]

# 多轮对话数据格式
[{
    "conversation":[
        {
            "system": "xxx",
            "input": "xxx",
            "output": "xxx"
        },
        {
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
        },
        {
            "input": "xxx",
            "output": "xxx"
        }
    ]
}]
```

## 准备和修改配置文件

```Bash
# 列出所有内置配置
xtuner list-cfg | grep "internlm2_chat_7b"
internlm2_chat_7b_full_finetune_custom_dataset_e1
internlm2_chat_7b_qlora_alpaca_e3
internlm2_chat_7b_qlora_code_alpaca_e3
internlm2_chat_7b_qlora_lawyer_e3
internlm2_chat_7b_qlora_oasst1_512_e3
internlm2_chat_7b_qlora_oasst1_e3
llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_finetune
llava_internlm2_chat_7b_clip_vit_large_p14_336_e1_gpu8_pretrain
llava_internlm2_chat_7b_qlora_clip_vit_large_p14_336_lora_e1_gpu8_finetune

cd ~/solomon
xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 .
```

### qlora

```Bash
# 修改配置文件
# 改个文件名
cp internlm2_chat_7b_qlora_oasst1_e3_copy.py internlm2_chat_7b_qlora_solomon_e3_copy.py

vim internlm2_chat_7b_qlora_solomon_e3_copy.py
```

减号代表要删除的行，加号代表要增加的行。

```Bash
# 单个文件情况：
# 修改import部分
- from xtuner.dataset.map_fns import oasst1_map_fn, template_map_fn_factory
+ from xtuner.dataset.map_fns import template_map_fn_factory

# 修改模型为本地路径
- pretrained_model_name_or_path = 'internlm/internlm-chat-7b'
+ pretrained_model_name_or_path = '/root/solomon/internlm2-chat-7b'

# 修改训练数据为 MedQA2019-structured-train.jsonl 路径
- data_path = 'timdettmers/openassistant-guanaco'
+ data_path = '/root/solomon/data/train_data/Aristotle_qlora.json'

# 原始2400条数据，保证总训练数据在2万条以上，2400*10=2.4万
- max_epochs= 3
+ max_epochs= 10
- batch_size = 1
+ batch_size = 1
- max_length = 2048
+ max_length = 2048
# 根据数据量调整，以免空间不足
- save_steps = 500
+ save_steps = 100
- save_total_limit = 2 # Maximum checkpoints to keep (-1 means unlimited)
+ save_total_limit = -1

# 用于评估输出内容的问题（用于评估的问题尽量与数据集的question保持一致）
evaluation_freq = 300
SYSTEM = '你是古希腊哲学家亚里士多德。你的目标:解答用户对于哲学思辨的疑问,以他的哲学思想及说话口吻进行专业的解答,拒绝回答与哲学问题无关的问题。'
evaluation_inputs = [
    '你好, 人生的终极价值体现在什么方面？', '请介绍一下你自己', '自我放纵的后果是什么？', '什么是罪恶的本质？'
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

### full_finetune（pretrain）

```Bash
xtuner copy-cfg internlm2_chat_7b_full_finetune_custom_dataset_e1 .
cp internlm2_chat_7b_full_finetune_custom_dataset_e1_copy.py internlm2_chat_7b_full_finetune_solomon_ds_e1_copy.py

vim internlm2_chat_7b_full_finetune_solomon_ds_e1_copy.py

+ from mmengine.config import read_base
- from xtuner.dataset.map_fns import template_map_fn_factory
- from xtuner.engine import (DatasetInfoHook, EvaluateChatHook, ThroughputHook,
                           VarlenAttnArgsToMessageHubHook)
+ from xtuner.engine import DatasetInfoHook

+ with read_base():
+    from .map_fn import single_turn_map_fn as dataset_map_fn

# PART 1  Settings
- pretrained_model_name_or_path = 'internlm/internlm2-chat-7b'
+ pretrained_model_name_or_path = '/root/solomon/internlm2-chat-7b'
- data_files = ['/path/to/json/file.json']
+ data_path = './Aristotle_doc.json'
+ data_files = ['./Aristotle_doc.json']
- prompt_template = PROMPT_TEMPLATE.internlm2_chat

- evaluation_inputs = [
        '请给我介绍五个上海的景点', 'Please tell me five scenic spots in Shanghai'
    ]
+ evaluation_inputs = [
        '你好, 人生的终极价值体现在什么方面？', '请介绍一下你自己', '自我放纵的后果是什么？', '什么是罪恶的本质？'
]

# PART 3  Dataset & Dataloader
- dataset_map_fn=None,
+ dataset_map_fn=dataset_map_fn,
- template_map_fn=dict(
            type=template_map_fn_factory, template=prompt_template),
+ template_map_fn=None,

# PART 5  Runtime
- custom_hooks = [
        dict(
            type=DatasetInfoHook, tokenizer=tokenizer,
            is_intern_repo_dataset=True),
        dict(
            type=EvaluateChatHook,
            tokenizer=tokenizer,
            every_n_iters=evaluation_freq,
            evaluation_inputs=evaluation_inputs,
            system=SYSTEM,
            prompt_template=prompt_template),
        dict(type=ThroughputHook)
    ]
+ custom_hooks = [dict(type=DatasetInfoHook, tokenizer=tokenizer)]
```

## 微调

### pretrain

xtuner train /root/solomon/internlm2_chat_7b_full_finetune_solomon_ds_e1_copy.py --deepspeed deepspeed_zero2

tip：调测失败，官方样例不适用新版xtune和interlm2的，后续有时间再调整

### qlora微调
```Bash
# 单卡
cd ~/solomon/
xtuner train /root/solomon/internlm2_chat_7b_qlora_solomon_e3_copy.py --deepspeed deepspeed_zero2

# 多卡
(DIST) NPROC_PER_NODE=${GPU_NUM} xtuner train /root/ft-Oculi/internlm2_chat_7b_qlora_Oculi_e3_copy.py --deepspeed deepspeed_zero2
(SLURM) srun ${SRUN_ARGS} xtuner train internlm2_chat_7b_qlora_oasst1_e3 --launcher slurm --deepspeed deepspeed_zero2

# --deepspeed deepspeed_zero2, 开启 deepspeed 加速
```

<img width="283" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/08f70245-d944-4c14-b5f9-672be8578dcb">

将保存的 PTH 模型（如果使用的DeepSpeed，则将会是一个文件夹）转换为 HuggingFace 模型，即：生成 Adapter 文件夹

```Bash
cd ~/solomon
mkdir hf_solomon
# 设置环境变量
export MKL_SERVICE_FORCE_INTEL=1

# xtuner convert pth_to_hf ${CONFIG_NAME_OR_PATH} ${PTH} ${SAVE_PATH}
xtuner convert pth_to_hf internlm2_chat_7b_qlora_solomon_e3_copy.py /root/solomon/work_dirs/internlm2_chat_7b_qlora_solomon_e3_copy/iter_1670.pth /root/solomon/hf_solomon
```

## 测试对话,分别测试哪个批次没有过拟合，效果较好

```Bash
# xtuner chat ${NAME_OR_PATH_TO_LLM} --adapter {NAME_OR_PATH_TO_ADAPTER} [optional arguments]
# 与 InternLM2-Chat-7B, hf_solomon(调用adapter_config.json) 对话：
cd ~/solomon
xtuner chat /root/solomon/internlm2-chat-7b --adapter /root/solomon/hf_solomon --prompt-template internlm2_chat
```

使用iter_500.pth的结果：

<img width="790" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/84b90760-c571-4e19-95f6-be762e308a0e">

使用iter_1000.pth的结果：

<img width="708" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/9b7ad39f-2903-47bd-a06d-ee69170c50c4">

使用iter_1670.pth的结果：

<img width="634" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/9125d652-218f-41f2-ae81-13ed31c76376">

结论：1670过拟合了，自己给起了一个名字，500没有效果，最好的是1000的，后面有时间调小一下save_steps

## 合并与测试

### 将 HuggingFace adapter 合并到大语言模型：

```Bash
cd ~/solomon
xtuner convert merge ./internlm2-chat-7b ./hf_solomon_1000 ./merged_solomon_1000 --max-shard-size 2GB
# xtuner convert merge \
#     ${NAME_OR_PATH_TO_LLM} \
#     ${NAME_OR_PATH_TO_ADAPTER} \
#     ${SAVE_PATH} \
#     --max-shard-size 2GB
```

<img width="225" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/5de45173-77bd-404e-b50d-deaaa0f05d19">

### 测试与合并后的模型对话

```Bash
# 加载 Adapter 模型对话（Float 16）

# xtuner chat ./merged_solomon --prompt-template internlm2_chat

# 4 bit 量化加载
xtuner chat ./merged_solomon_1000 --bits 4 --prompt-template internlm2_chat
```

<img width="637" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/5f09d75f-a3cb-4181-8205-0a814ca3df66">

<img width="665" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/b3370a7e-6f82-47eb-b757-e8df5cff3ccc">


# WEB_Demo

```Bash
# 创建code文件夹用于存放InternLM项目代码
cd ~
mkdir code && cd code
git clone https://github.com/InternLM/InternLM.git

cd ~/code
cp ~/code/chat/web_demo.py web_solomon.py

vim web_solomon.py
# 修改将 code/web_solomon.py 中 183 行和 186 行的模型路径更换为Merge后存放参数的路径 /root/solomon/merged_solomon
+ model = (AutoModelForCausalLM.from_pretrained('/root/solomon/merged_solomon_1000',
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
+ tokenizer = AutoTokenizer.from_pretrained('/root/solomon/merged_solomon_1000',
                                              trust_remote_code=True)
#216行
- meta_instruction = ('You are InternLM (书生·浦语), a helpful, honest, '
                        'and harmless AI assistant developed by Shanghai '
                        'AI Laboratory (上海人工智能实验室).')
+ meta_instruction = ('你是古希腊哲学家亚里士多德，请以他的哲学思想和口吻回答问题。')
# 修改239 行和 240 行
+ user_avator = '/root/code/InternLM/assets/user.png'
+ robot_avator = '/root/code/data/Aristotle.png'
+ st.title('与古希腊哲学家思辨')

pip install streamlit==v1.31.1

streamlit run /root/code/web_solomon.py --server.address 127.0.0.1 --server.port 6006

# 本地运行
ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p 37660(修改对应端口)
浏览器访问：http://127.0.0.1:6006
```

<img width="829" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/4b8f2d71-7f11-4d39-a3a5-55de40846828">

# 量化模型

```Bash
# requirements
python 3.8+
lmdeploy
torch<=2.1.2,>=2.0.0
transformers>=4.33.0,<=4.38.1
triton>=2.1.0,<=2.2.0
gradio<4.0.0
```

## 安装 lmdeploy

```Bash
# 解决 ModuleNotFoundError: No module named 'packaging' 问题
pip install packaging
# 使用 flash_attn 的预编译包解决安装过慢问题
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

# pip install 'lmdeploy[all]==v0.1.0'
pip install lmdeploy=0.2.5
```

离线转换

```Bash
# 转换模型（FastTransformer格式） 把 huggingface 格式的模型，转成 turbomind 推理格式，得到一个 workspace 目录
# 转换模型的layout，存放在默认路径 ./workspace 下
lmdeploy convert internlm2-chat-7b /root/solomon/merged_solomon_1000/
```

<img width="791" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/650e01ec-38c8-4b36-a922-57f9be31e0ab">

执行完成后将会在当前目录生成一个 workspace 的文件夹。这里面包含的就是 TurboMind 和 Triton “模型推理”需要到的文件。

<img width="425" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/4c153fc3-713c-4586-8289-3ef8ca7e8404">

weights 和 tokenizer 目录分别放的是拆分后的参数和 Tokenizer

每一份参数第一个 0 表示“层”的索引，后面的那个0表示 Tensor 并行的索引，因为我们只有一张卡，所以被拆分成 1 份。如果有两张卡可以用来推理，则会生成0和1两份，也就是说，会把同一个参数拆成两份。比如 layers.0.attention.w_qkv.0.weight 会变成 layers.0.attention.w_qkv.0.weight 和 layers.0.attention.w_qkv.1.weight。执行 lmdeploy convert 命令时，可以通过 --tp 指定（tp 表示 tensor parallel），该参数默认值为1（也就是一张卡）

<img width="371" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/41646fd8-557c-4e4d-aa89-dc89250ce891">

## 开启 KV Cache INT8 量化

(当显存不足，或序列比较长时)

kv cache PTQ 量化，使用的公式如下：

```Bash
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```

第一步执行命令获取量化参数，并保存至quant_output目录

```Bash
export HF_MODEL=/root/solomon/merged_solomon_1000/

# 计算 minmax
lmdeploy lite calibrate \
  $HF_MODEL \
  --calib-dataset 'ptb' \
  --calib-samples 64 \
  --calib-seqlen 1024 \
  --work-dir $HF_MODEL
```

这个命令行中选择 128 条输入样本，每条样本长度为 2048，数据集选择 ptb，输入模型后就会得到上面的各种统计值。

如果显存不足，可以适当调小 samples 的数量或 sample 的长度。

<img width="418" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/3134fd0c-f7fe-4615-b809-98192995b44b">

<img width="225" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/5de45173-77bd-404e-b50d-deaaa0f05d19">
<img width="225" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/4997aab4-bc42-4a9f-8800-4510d394f052">

<img width="124" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/29306a6d-4207-4a44-9554-202514f0253f">

对比原来多了4个文件。

测试聊天效果。注意需要添加参数--quant-policy 4以开启KV Cache int8模式。

```Bash
lmdeploy chat turbomind $HF_MODEL --model-format hf --quant-policy 4
```

<img width="724" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/b6db7071-cec1-4f84-b671-2d4758ac8238">

1/4的A100,20G显存爆了，调小 samples 的数量和sample 的长度，试了还是不行

```Bash
lmdeploy lite calibrate \
  $HF_MODEL \
  --calib-dataset 'ptb' \
  --calib-samples 64 \
  --calib-seqlen 1024 \
  --work-dir $HF_MODEL
```

<img width="371" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/988d2b45-0324-4ebc-9458-b788d9ea7ec5">

## w4a16

没有 GPU 卡，只有 CPU，尝试量化版本。W4A16中的A是指Activation，保持FP16，只对参数进行 4bit 量化。

量化结束后，权重文件存放在 $WORK_DIR 下

```Bash
# LMDeploy 使用 AWQ 算法，实现模型 4bit 权重量化
export HF_MODEL=/root/solomon/merged_solomon_1000/
export WORK_DIR=/root/solomon/merged_solomon_1000-4bit/
pip install fuzzywuzzy

# 量化权重模型
# w_bits 表示量化的位数，w_group_size 表示量化分组统计的尺寸，work_dir 是量化后模型输出的位置。
lmdeploy lite auto_awq \
   $HF_MODEL \                       # Model name or path, either model repo name on huggingface hub like 'internlm/internlm-chat-7b', or a model path in local host
  --calib-dataset 'ptb' \            # Calibration dataset, supports c4, ptb, wikitext2, pileval
  --calib-samples 128 \              # Number of samples in the calibration set, if memory is insufficient, you can appropriately reduce this
  --calib-seqlen 2048 \              # Length of a single piece of text, if memory is insufficient, you can appropriately reduce this
  --w-bits 4 \                       # Bit number for weight quantization
  --w-group-size 128 \               # Group size for weight quantization statistics
  --work-dir $WORK_DIR               # Folder storing Pytorch format quantization statistics parameters and post-quantization weight

lmdeploy lite auto_awq \
   $HF_MODEL \
  --calib-dataset 'ptb' \
  --calib-samples 128 \
  --calib-seqlen 2048 \
  --w-bits 4 \
  --w-group-size 128 \
  --work-dir $WORK_DIR
```

<img width="377" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/2c2e8d74-327e-4716-a32c-c0c7ee18aa9c">

<img width="207" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/e0002754-60bd-45da-bea4-7ac2bb59b3f1">

测试INT4 模型量化效果

```Bash
# 直接在控制台和模型对话
lmdeploy chat turbomind ./merged_solomon_1000-4bit --model-format awq

<img width="255" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/73267d5b-3965-458f-9062-b5e945078d3f">

<img width="758" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/8a9338e7-e641-4e94-b185-f49f241114a6">
```

```Bash
# 直接对internlm2-chat-7b量化测试
export HF_MODEL=/root/solomon/internlm2-chat-7b
export WORK_DIR=/root/solomon/internlm2-chat-7b-4bit

lmdeploy chat turbomind ./internlm2-chat-7b-4bit --model-format awq
```

同样报错，看来InternLM2量化还不行

<img width="767" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/a496cd4a-3fbe-4b46-a163-67c31e97d8cf">


```Bash
# kCacheKVInt8 和 WeightInt4 两种方案可以同时开启
lmdeploy chat turbomind ./merged_solomon_1000-4bit --model-format awq --quant-policy 4
# 启动gradio服务
lmdeploy serve gradio ./merged_solomon_1000-4bit --server-name {ip_addr} --server-port {port} --model-format awq
在浏览器中打开 http://{ip_addr}:{port}，即可在线对话
```

# 模型上传和部署openxlab

## 模型上传准备工作

打开 InternLM2-chat-7b在openxlab上的模型链接，切换到 模型文件-> 点击查看元信息：

> https://openxlab.org.cn/models/detail/OpenLMLab/internlm2-chat-7b

<img width="783" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/40685221-71ed-498c-9c8c-e0ddc77cb1c3">

cd ~/solomon/merged_solomon_1000

新建metafile.yml, 将里面的内容复制到 metafile.yml文件中

```Bash
Collections:
- Name: "与古希腊哲学家思辨"
  License: "Apache-2.0"
  Framework: "[]"
  Paper: {}
  Code:
    URL: "https://github.com/superkong001/InternLM_project/solomon_chart"
Models:
- Name: "config.json"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "configuration_internlm2.py"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "generation_config.json"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "modeling_internlm2.py"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00001-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00002-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00003-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00004-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00005-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00006-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00007-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model-00008-of-00008.bin"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "special_tokens_map.json"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "tokenization_internlm2.py"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "tokenizer_config.json"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "tokenizer.model"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
- Name: "pytorch_model.bin.index.json"
  Results:
  - Task: "Text Generation"
    Dataset: "none"
```

pip install ruamel.yaml

编辑 convert.py

```Bash
import sys
import ruamel.yaml

yaml = ruamel.yaml.YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False
file_path = 'metafile.yml'
# 读取YAML文件内容
with open(file_path, 'r') as file:
 data = yaml.load(file)
# 遍历模型列表
for model in data.get('Models', []):
 # 为每个模型添加Weights键值对，确保名称被正确引用
 model['Weights'] = model['Name']

# 将修改后的数据写回文件
with open(file_path, 'w') as file:
 yaml.dump(data, file)

print("Modifications saved to the file.")
```

生成好带weight的文件：

python convert.py metafile.yml

<img width="496" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/b7583325-4016-4f86-a078-930ae4051ca3">

打开 openxlab右上角 账号与安全--》密钥管理:

<img width="857" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/a8d92d0a-0be5-439d-bb78-164ca98aadf2">

将AK,SK复制下来。

配置登录信息：

```Bash
pip install openxlab
python
import openxlab
openxlab.login(ak='xxx',sk='yyyy')
```

创建并上传模型：

openxlab model create --model-repo='superkong001/solomon_chart' -s ./metafile.yml

Tips：漏改的话继续上传，新建并编辑一个upload1.yml

```Bash
# 部分上传：
python
from openxlab.model import upload 
upload(model_repo='superkong001/solomon_chart', file_type='metafile', source="upload1.yml")

# 全量更新：
python
from openxlab.model import upload
upload(model_repo='superkong001/solomon_chart', file_type='metafile', source="metafile.yml")

```

<img width="812" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/f173bd93-4ea7-4648-ab9e-9053a18b51f4">

上传后的模型：

<img width="656" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/4d0b358b-4094-4bd3-b8b3-09984c1e1501">

下载web_solomon.py，并修改保存为solomon_Webchart.py：

```Bash
+ from modelscope import snapshot_download
+ from openxlab.model import download

# 修改load_model
- def load_model():
    model = (AutoModelForCausalLM.from_pretrained('/root/solomon/merged_solomon_1000',
                                                  trust_remote_code=True).to(
                                                      torch.bfloat16).cuda())
    tokenizer = AutoTokenizer.from_pretrained('/root/solomon/merged_solomon_1000',
                                              trust_remote_code=True)
    return model, tokenizer

# 改为：
+ def load_model():
    # 定义模型路径(modelscope)
    from modelscope import snapshot_download
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    model_id = "teloskong/solomon_chart"
    model = (
        AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        .to(torch.bfloat16)
        .cuda()
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    model_path = snapshot_download(model_id, revision='master')

    # 定义模型路径(xlab)
    # from openxlab.model import download
    # from transformers import AutoTokenizer, AutoModelForCausalLM
    # model_id = 'telos/solomon_chart'
    # model_name = 'solomon_chart'
    # # model_path = '/home/xlab-app-center/.cache' # '/home/xlab-app-center/.cache/model'
    # model_path = './'
    # download(model_repo=model_id, model_name=model_name)
    
    # # 从预训练的模型中获取模型，并设置模型参数
    # model = (AutoModelForCausalLM.from_pretrained(model_path,
    #                                               trust_remote_code=True).to(
    #                                                   torch.bfloat16).cuda())
    
    # # 从预训练的模型中获取tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_path,
    #                                           trust_remote_code=True)

    
    # model.eval()  
    
    return model, tokenizer, model_path

# 修改combine_history：
+ meta_instruction = ('你是古希腊哲学家亚里士多德，请以他的哲学思想和口吻回答问题。你的目标:解答用户对于哲学思辨的疑问,以他的哲学思想及说话口吻进行专业的解答,拒绝回答与哲学问题无关的问题。')

# 修改main函数
+ model, tokenizer, mode_name_or_path = load_model()
+ user_avator = './user.png'
+ robot_avator = './Aristotle.png'
+ st.title('InternLM2-Chat-7B 亚里士多德')
```

# 模型上传modelscope

在modelscope创建模型solomon_chart：

<img width="727" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/af8002c2-38d5-4060-b597-dc3bd8391a27">

```Bash
mkdir ~/modelscope
cd ~/modelscope
apt-get install git-lfs
# 需要公开后才能下载
git clone https://www.modelscope.cn/teloskong/solomon_chart.git

# 将 /root/solomon/merged_solomon_1000 模型文件覆盖 ~/modelscope/solomon_chart 下的文件
cd solomon_chart/
cp -r /root/solomon/merged_solomon_1000/* .

# 上传模型文件
git add *
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git commit -m "solomon_chart Model V20240308"
git push # 输入用户名和Git 访问令牌
```

<img width="742" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/b929b0a2-8ab4-486d-a668-2f21f6646156">

## openxlab部署

创建 app.py 添加至代码仓库

```Bash
import os

if __name__ == '__main__':
    os.system('streamlit run solomon_Webchart.py --server.address 0.0.0.0 --server.port 7860 --server.enableStaticServing True')
```

创建requirements.txt

```Bash
pandas
torch
torchvision
modelscope
transformers
xtuner
streamlit
openxlab
```

<img width="449" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/9b82645f-55b1-443c-9915-b5d2ced6a549">

<img width="650" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/23c0951a-dc09-4f8f-a744-252625da1400">

<img width="860" alt="image" src="https://github.com/superkong001/InternLM_project/assets/37318654/9b03207c-348a-40ef-a49d-3247106c4048">

<img width="794" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/c8ca2572-2503-4dc0-86b0-d9439fa060ad">

<img width="565" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/4f734220-c300-4697-bc30-435d4de6ac9d">


