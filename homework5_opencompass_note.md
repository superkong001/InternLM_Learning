<img width="1038" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/76ec7196-a7c7-4093-b0d0-d3da42df24c5">

为什么需要进行评测：

<img width="1075" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/20a42d37-11e8-4ca8-b612-ce952d8d0d3a">

<img width="545" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/6fc0b7f6-e8e7-4f16-9b38-bc26e1b878a1">

## 大模型评测方法

<img width="1026" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/c85d1388-1c81-47e2-9c2d-293cb350cdd5">

斯坦福大学提出了较为系统的评测框架HELM，从准确性，安全性，鲁棒性和公平性等维度开展模型评测。

纽约大学联合谷歌和Meta提出了SuperGLUE评测集，从推理能力，常识理解，问答能力等方面入手，构建了包括8个子任务的大语言模型评测数据集。

加州大学伯克利分校提出了MMLU测试集，构建了涵盖高中和大学的多项考试，来评估模型的知识能力和推理能力。

谷歌也提出了包含数理科学，编程代码，阅读理解，逻辑推理等子任务的评测集Big-Bench，涵盖200多个子任务，对模型能力进行系统化的评估。

在中文评测方面，国内的学术机构也提出了如CLUE,CUGE等评测数据集，从文本分类，阅读理解，逻辑推理等方面评测语言模型的中文能力。

OpenCompass提供设计一套全面、高效、可拓展的大模型评测方案，对模型能力、性能、安全性等进行全方位的评估。
OpenCompass提供分布式自动化的评测系统，支持对(语言/多模态)大模型开展全面系统的能力评估。

## OpenCompass介绍

### 评测对象

基座模型：一般是经过海量的文本数据以自监督学习的方式进行训练获得的模型（如OpenAI的GPT-3，Meta的LLaMA），往往具有强大的文字续写能力。

对话模型：一般是在的基座模型的基础上，经过指令微调或人类偏好对齐获得的模型（如OpenAI的ChatGPT、上海人工智能实验室的书生·浦语），能理解人类指令，具有较强的对话能力。

<img width="1010" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/21fc0351-7186-4be5-a73b-97e8b367e5f1">

### 工具架构

![image](https://github.com/superkong001/InternLM_Learning/assets/37318654/f5395554-5087-4b89-a2e5-db5d0c311e94)

模型层：大模型评测所涉及的主要模型种类，OpenCompass以基座模型和对话模型作为重点评测对象。

能力层：OpenCompass从通用能力和特色能力两个方面来进行评测维度设计。
在通用能力方面，从语言、知识、理解、推理、安全等多个能力维度进行评测。
在特色能力方面，从长文本、代码、工具、知识增强等维度进行评测。

方法层：OpenCompass采用客观评测与主观评测两种评测方式。
客观评测能便捷地评估模型在具有确定答案（如选择，填空，封闭式问答等）的任务上的能力；主观评测能评估用户对模型回复的真实满意度。
OpenCompass采用基于模型辅助的主观评测和基于人类反馈的主观评测两种方式。

工具层：OpenCompass提供丰富的功能支持自动化地开展大语言模型的高效评测。包括分布式评测技术，提示词工程，对接评测数据库，评测榜单发布，评测报告生成等诸多功能。

## OpenCompass评测方法

OpenCompass采取客观评测与主观评测相结合的方法。针对具有确定性答案的能力维度和场景，通过构造丰富完善的评测集，对模型能力进行综合评价。针对体现模型能力的开放式或半开放式的问题、模型安全问题等，采用主客观相结合的评测方式。

### 客观评测

<img width="607" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/2257799e-b279-49fb-a3cb-c13624ffab1a">

针对具有标准答案的客观问题，通过使用定量指标比较模型的输出与标准答案的差异，并根据结果衡量模型的性能。同时，由于大语言模型输出自由度较高，在评测阶段，需要对其输入和输出作一定的规范和设计，尽可能减少噪声输出在评测阶段的影响，才能对模型的能力有更加完整和客观的评价。

为了更好地激发出模型在题目测试领域的能力，并引导模型按照一定的模板输出答案，OpenCompass采用提示词工程 （prompt engineering）和语境学习（in-context learning）进行客观评测。

在客观评测的具体实践中，通常采用下列两种方式进行模型输出结果的评测：

判别式评测：该评测方式基于将问题与候选答案组合在一起，计算模型在所有组合上的困惑度（perplexity），并选择困惑度最小的答案作为模型的最终输出。例如，若模型在问题?答案1上的困惑度为0.1，在问题?答案2上的困惑度为0.2，最终会选择答案1作为模型的输出。

生成式评测：该评测方式主要用于生成类任务，如语言翻译、程序生成、逻辑分析题等。具体实践时，使用问题作为模型的原始输入，并留白答案区域待模型进行后续补全。通常还需要对其输出进行后处理，以保证输出满足数据集的要求。

<img width="962" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/fa79d726-afd9-4631-a0ec-1e3728b0e172">

### 主观评测

<img width="892" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/d046a4b9-dd63-4f3f-b603-4f54cfc1e111">

语言表达生动精彩，变化丰富，大量的场景和能力无法凭借客观指标进行评测。针对如：模型安全和模型语言能力的评测，以人的主观感受为主的评测更能体现模型的真实能力，并更符合大模型的实际使用场景。

OpenCompass采取的主观评测方案是指借助受试者的主观判断对具有对话能力的大语言模型进行能力评测。在具体实践中，提前基于模型的能力维度构建主观测试问题集合，并将不同模型对于同一问题的不同回复展现给受试者，收集受试者基于主观感受的评分。由于主观测试成本高昂，本方案同时也采用使用性能优异的大语言模拟人类进行主观打分。在实际评测中，本文将采用真实人类专家的主观评测与基于模型打分的主观评测相结合的方式开展模型能力评估。

在具体开展主观评测时，OpenComapss采用单模型回复满意度统计和多模型满意度比较两种方式开展具体的评测工作。

## OpenCompass工作流程

![image](https://github.com/superkong001/InternLM_Learning/assets/37318654/c3cfefd2-33ac-4bee-b1c4-1dc3c073352f)

在 OpenCompass 中评估一个模型通常包括以下几个阶段：配置 -> 推理 -> 评估 -> 可视化。

配置：这是整个工作流的起点。需要配置整个评估过程，选择要评估的模型和数据集。此外，还可以选择评估策略、计算后端等，并定义显示结果的方式。

推理与评估：在这个阶段，OpenCompass 将会开始对模型和数据集进行并行推理和评估。推理阶段主要是让模型从数据集产生输出，而评估阶段则是衡量这些输出与标准答案的匹配程度。这两个过程会被拆分为多个同时运行的“任务”以提高效率，但注意，如果计算资源有限，这种策略可能会使评测变得更慢。

可视化：评估完成后，OpenCompass 将结果整理成易读的表格，并将其保存为 CSV 和 TXT 文件。也可以激活飞书状态上报功能，此后可以在飞书客户端中及时获得评测状态报告。

通过 OpenCompass 展示书生浦语在 [C-Eval](https://cevalbenchmark.com/index.html#home) 基准任务上的评估。
配置文件可以在 [configs/eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py) 中找到。

## 其他评测

多模态评测 MMBench

<img width="1079" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/afdf33e0-b430-48f8-abf2-3429eea7098a">

垂直领域评测

<img width="1043" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/4845a4a4-9107-46c5-b996-8c2d652b186b">


# 实操

## 安装

### 面向GPU的环境安装

```bash
conda create --name opencompass --clone=/root/share/conda_envs/internlm-base
source activate opencompass
git clone https://github.com/open-compass/opencompass
cd opencompass
pip install -e .
```

## 数据准备

```bash
# 解压评测数据集到 data/ 处
cp /share/temp/datasets/OpenCompassData-core-20231110.zip /root/opencompass/
unzip OpenCompassData-core-20231110.zip

# 将会在opencompass下看到data文件夹
```

查看支持的数据集和模型：

```bash
# 列出所有跟 internlm 及 ceval 相关的配置
python tools/list_configs.py internlm ceval
```

## 客观评测操作

<img width="688" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/0f4bfda1-370c-4517-aff3-477923f161b0">

```bash
python run.py --datasets ceval_gen --hf-path /share/temp/model_repos/internlm-chat-7b/ --tokenizer-path /share/temp/model_repos/internlm-chat-7b/ --tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True --model-kwargs trust_remote_code=True device_map='auto' --max-seq-len 2048 --max-out-len 16 --batch-size 4 --num-gpus 1 --debug
```

命令解析：

```bash
--datasets ceval_gen \
--hf-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace 模型路径
--tokenizer-path /share/temp/model_repos/internlm-chat-7b/ \  # HuggingFace tokenizer 路径（如果与模型路径相同，可以省略）
--tokenizer-kwargs padding_side='left' truncation='left' trust_remote_code=True \  # 构建 tokenizer 的参数
--model-kwargs device_map='auto' trust_remote_code=True \  # 构建模型的参数
--max-seq-len 2048 \  # 模型可以接受的最大序列长度
--max-out-len 16 \  # 生成的最大 token 数
--batch-size 4  \  # 批量大小
--num-gpus 1  # 运行模型所需的 GPU 数量
--debug
```

有关 `run.py` 支持的所有与 HuggingFace 相关的参数，可以阅读 [评测任务发起](https://opencompass.readthedocs.io/zh-cn/latest/user_guides/experimentation.html#id2)

除了通过命令行配置实验外，OpenCompass 还允许用户在配置文件中编写实验的完整配置，并通过 `run.py` 直接运行它。配置文件是以 Python 格式组织的，并且必须包括 `datasets` 和 `models` 字段。

示例测试配置在 [configs/eval_demo.py](https://github.com/open-compass/opencompass/blob/main/configs/eval_demo.py) 中。此配置通过 [继承机制](../user_guides/config.md#继承机制) 引入所需的数据集和模型配置，并以所需格式组合 `datasets` 和 `models` 字段。

```bash
from mmengine.config import read_base

with read_base():
    from .datasets.siqa.siqa_gen import siqa_datasets
    from .datasets.winograd.winograd_ppl import winograd_datasets
    from .models.opt.hf_opt_125m import opt125m
    from .models.opt.hf_opt_350m import opt350m

datasets = [*siqa_datasets, *winograd_datasets]
models = [opt125m, opt350m]
```

脚本运行：

```bash
python run.py configs/eval_demo.py
```

OpenCompass 提供了一系列预定义的模型配置，位于 configs/models 下

# 使用 `HuggingFaceCausalLM` 评估由 HuggingFace 的 `AutoModelForCausalLM` 支持的模型
from opencompass.models import HuggingFaceCausalLM

```bash
# OPT-350M
opt350m = dict(
       type=HuggingFaceCausalLM,
       # `HuggingFaceCausalLM` 的初始化参数
       path='facebook/opt-350m',
       tokenizer_path='facebook/opt-350m',
       tokenizer_kwargs=dict(
           padding_side='left',
           truncation_side='left',
           proxies=None,
           trust_remote_code=True),
       model_kwargs=dict(device_map='auto'),
       # 下面是所有模型的共同参数，不特定于 HuggingFaceCausalLM
       abbr='opt350m',               # 结果显示的模型缩写
       max_seq_len=2048,             # 整个序列的最大长度
       max_out_len=100,              # 生成的最大 token 数
       batch_size=64,                # 批量大小
       run_cfg=dict(num_gpus=1),     # 该模型所需的 GPU 数量
    )
```

使用配置时，可以通过命令行参数 --models 指定相关文件，或使用继承机制将模型配置导入到配置文件中的 models 列表中。

与模型类似，数据集的配置文件也提供在 configs/datasets 下。用户可以在命令行中使用 --datasets，或通过继承在配置文件中导入相关配置

下面是来自 configs/eval_demo.py 的与数据集相关的配置片段：

```bash
from mmengine.config import read_base  # 使用 mmengine.read_base() 读取基本配置

with read_base():
    # 直接从预设的数据集配置中读取所需的数据集配置
    from .datasets.winograd.winograd_ppl import winograd_datasets  # 读取 Winograd 配置，基于 PPL（困惑度）进行评估
    from .datasets.siqa.siqa_gen import siqa_datasets  # 读取 SIQA 配置，基于生成进行评估

datasets = [*siqa_datasets, *winograd_datasets]       # 最终的配置需要包含所需的评估数据集列表 'datasets'
```

数据集配置通常有两种类型：'ppl' 和 'gen'，分别指示使用的评估方法。其中 ppl 表示辨别性评估，gen 表示生成性评估。

此外，[configs/datasets/collections](https://github.com/open-compass/opencompass/blob/main/configs/datasets/collections) 收录了各种数据集集合，方便进行综合评估。OpenCompass 通常使用 [`base_medium.py`](https://github.com/open-compass/opencompass/blob/main/configs/datasets/collections/base_medium.py) 进行全面的模型测试。要复制结果，只需导入该文件，例如：

```bash
python run.py --models hf_llama_7b --datasets base_medium
```

## 主观评测

### Inferen Stage
修改需要评测模型

<img width="927" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/6ba6062c-4210-4a68-8b6e-27a640b91edf">

修改模型和Tokenizer路径（从huggingface改成本地）&修改输出长度max_out_len（dataset里指定了话会自动覆盖）

<img width="824" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/696aa8ce-b125-4d07-94b0-956b65e20829">

<img width="802" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/12459529-7850-4b18-b89d-d4c804dce091">

开sample list，防止失去模型回复多样性

<img width="617" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/a1c86b78-05ad-497f-b733-d7b37fb17f92">

Tips: 在集群上分片同时推理：

<img width="290" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/8421476a-ed0f-4c99-aa82-e12f2ca9a412">

在单机上推理：

<img width="584" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/110d94d9-3543-4640-8508-190e86147b6a">

通过config运行示例：

<img width="697" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/a696e8c6-70bd-4804-bfba-5870f22af758">

--debug 在终端里打印过程

--reuse latest 从最新时间戳下继续运行（断点继续） 

<img width="682" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/3131385c-5f98-4b78-a3ea-97f14de224d1">

### Judge Model

可以替换需要的judge model，参考对应model里的写法，或者import

<img width="902" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/ba9b7d27-a004-4995-86e7-186245db05a0">

<img width="636" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/860b00b1-617a-4d3d-b727-2946cc66cac4">

### Evaluation Configuration

type：设置是否分片，参考inferen

mode：singlescore 打分模式、m2n对战模式。。。

summarizer：指定汇总评测结果方式

# 实操记录




