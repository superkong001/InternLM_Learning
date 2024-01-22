# 理论

![14b3d7982463a4b7486ba141f613bdf2_lmdeploy drawio](https://github.com/superkong001/InternLM_Learning/assets/37318654/854bac23-b633-44c3-b092-f083b125544c)

接下来我们切换到刚刚的终端（就是上图右边的那个「bash」，下面的「watch」就是监控的终端），创建部署和量化需要的环境。建议大家使用官方提供的环境，使用 conda 直接复制。

这里 `/share/conda_envs` 目录下的环境是官方未大家准备好的基础环境，因为该目录是共享只读的，而我们后面需要在此基础上安装新的软件包，所以需要复制到我们自己的 conda 环境（该环境下我们是可写的）。

7B, 14亿参数，FP16半精度占2个字节，估算需要14G内存

<img width="601" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/99bb7947-1a50-4152-aae5-139459ce6000">

token by token，decode-only, 所以需要保存历史对话，那KV会多

<img width="592" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/0a8ae3cf-e4c3-45e9-92b8-939861ccd622">

LMDeploy简介：

<img width="671" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/442ed71f-f405-437b-a985-83b5ec95accf">

## 量化

为什么做量化？

<img width="586" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/82434138-707b-47d6-9490-2e36669510d9">

<img width="668" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/1d76547c-2184-472d-b198-f405e23a2f5a">

<img width="665" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/4112f257-9bc7-4335-9c36-b9b20b4e07c7">

大模型推理性能优化：KV Cache（键值缓存） Int8，对中间键值（Key-Value）进行INT8量化

## 推理引擎TurboMind

<img width="539" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/012053d3-ef94-43da-b2c8-cd826b03ada1">

<img width="642" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/3adab746-376a-473c-94ed-22da04968db6">

<img width="343" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/e5c6fc10-4be2-46d7-981c-7feeb8a7a953">

<img width="662" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/6e2e93ed-17bc-40b7-b5e7-c8fdbb397845">

<img width="622" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/d136b462-d493-4e70-806b-8e003e555fea">

在 Transformer 架构中，注意力机制通常包括三个主要组件：Query（查询）、Key（键）和 Value（值）。

Query (Q)：查询是当前处理的输入部分，可以理解为模型正在试图理解或对其进行响应的部分。

Key (K)：键是存储在模型中的信息的一部分，用于与查询进行匹配。在处理输入时，模型会将查询与各个键进行比较，以确定输入的哪些部分最相关。

Value (V)：一旦确定了与查询最匹配的键，相应的值就会被用来构造输出。值包含与键相关联的实际信息，这些信息将被用来回答查询或对其做出反应。

在 Transformer 的注意力机制中，这些组件协同工作，使模型能够关注（即“注意”）输入中的不同部分，这对于理解复杂的语言结构至关重要。例如，在处理一个句子时，模型可能需要关注与当前词相关的其他词，而 Key-Value 对就是实现这种关注的机制。

<img width="647" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/66185c5e-fbae-4095-9144-810b05e631a2">

## 推理服务 API server

<img width="615" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/24899703-cb7a-4d81-a26a-5ee399785f9a">

**关于Tensor并行**

Tensor并行一般分为行并行或列并行，原理如下图所示。

![ef6f5d70abc499d710200fb51feddd20_6](https://github.com/superkong001/InternLM_Learning/assets/37318654/ad9784e7-3768-4b54-a305-9ee019d81823)

<p align="center">列并行<p>

![7386c840b50fe177be05495f83c2c4e9_7](https://github.com/superkong001/InternLM_Learning/assets/37318654/b493dc35-e0ab-476d-a6e3-2f55e40c6bc9)

<p align="center">行并行<p>

简单来说，就是把一个大的张量（参数）分到多张卡上，分别计算各部分的结果，然后再同步汇总。

# 实践相关

### 2.3 TurboMind推理+API服务

```bash
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1

# instance_num: batch slots
# tp: tensor 并行 
```
### 最佳实践

![a9f64eab292824a56d19dec0fdbc9c14_add4](https://github.com/superkong001/InternLM_Learning/assets/37318654/815b80e2-d63b-418b-9202-4f6667e705d4)

上面的性能对比包括两个场景：

场景一（前4张图）：固定的输入、输出 token 数（分别1和2048），测试Token输出吞吐量（output token throughput）。

场景二（第5张图）：使用真实数据，测试吞吐量（request throughput）。

场景一中，BatchSize=64时，TurboMind 的吞吐量超过 2000 token/s，整体比 DeepSpeed 提升约 5% - 15%；BatchSize=32时，比 Huggingface 的Transformers 提升约 3 倍；其他BatchSize时 TurboMind 也表现出优异的性能。

场景二中，对比了 TurboMind 和 vLLM 在真实数据上的吞吐量（request throughput）指标，TurboMind 的效率比 vLLM 高 30%

#### 2.6.2 模型配置实践

不知道大家还有没有印象，在离线转换（2.1.2）一节，我们查看了 `weights` 的目录，里面存放的是模型按层、按并行卡拆分的参数，不过还有一个文件 `config.ini` 并不是模型参数，它里面存的主要是模型相关的配置信息。下面是一个示例。

```ini
[llama]
model_name = internlm-chat-7b
tensor_para_size = 1
head_num = 32
kv_head_num = 32
vocab_size = 103168
num_layer = 32
inter_size = 11008
norm_eps = 1e-06
attn_bias = 0
start_id = 1
end_id = 2
session_len = 2056
weight_type = fp16
rotary_embedding = 128
rope_theta = 10000.0
size_per_head = 128
group_size = 0
max_batch_size = 64
max_context_token_num = 1
step_length = 1
cache_max_entry_count = 0.5
cache_block_seq_len = 128
cache_chunk_size = 1
use_context_fmha = 1
quant_policy = 0
max_position_embeddings = 2048
rope_scaling_factor = 0.0
use_logn_attn = 0
```

其中，模型属性相关的参数不可更改，主要包括下面这些。

```ini
model_name = llama2
head_num = 32
kv_head_num = 32
vocab_size = 103168
num_layer = 32
inter_size = 11008
norm_eps = 1e-06
attn_bias = 0
start_id = 1
end_id = 2
rotary_embedding = 128
rope_theta = 10000.0
size_per_head = 128
```

和数据类型相关的参数也不可更改，主要包括两个。

```ini
weight_type = fp16
group_size = 0
```

`weight_type` 表示权重的数据类型。目前支持 fp16 和 int4。int4 表示 4bit 权重。当 `weight_type` 为 4bit 权重时，`group_size` 表示 `awq` 量化权重时使用的 group 大小。

剩余参数包括下面几个。

```ini
tensor_para_size = 1
session_len = 2056
max_batch_size = 64
max_context_token_num = 1
step_length = 1
cache_max_entry_count = 0.5
cache_block_seq_len = 128
cache_chunk_size = 1
use_context_fmha = 1
quant_policy = 0
max_position_embeddings = 2048
rope_scaling_factor = 0.0
use_logn_attn = 0
```

一般情况下，我们并不需要对这些参数进行修改，但有时候为了满足特定需要，可能需要调整其中一部分配置值。这里主要介绍三个可能需要调整的参数。

- KV int8 开关：
    - 对应参数为 `quant_policy`，默认值为 0，表示不使用 KV Cache，如果需要开启，则将该参数设置为 4。
    - KV Cache 是对序列生成过程中的 K 和 V 进行量化，用以节省显存。我们下一部分会介绍具体的量化过程。
    - 当显存不足，或序列比较长时，建议打开此开关。
- 外推能力开关：
    - 对应参数为 `rope_scaling_factor`，默认值为 0.0，表示不具备外推能力，设置为 1.0，可以开启 RoPE 的 Dynamic NTK 功能，支持长文本推理。另外，`use_logn_attn` 参数表示 Attention 缩放，默认值为 0，如果要开启，可以将其改为 1。
    - 外推能力是指推理时上下文的长度超过训练时的最大长度时模型生成的能力。如果没有外推能力，当推理时上下文长度超过训练时的最大长度，效果会急剧下降。相反，则下降不那么明显，当然如果超出太多，效果也会下降的厉害。
    - 当推理文本非常长（明显超过了训练时的最大长度）时，建议开启外推能力。
- 批处理大小：
    - 对应参数为 `max_batch_size`，默认为 64，也就是我们在 API Server 启动时的 `instance_num` 参数。
    - 该参数值越大，吞度量越大（同时接受的请求数），但也会占用更多显存。
    - 建议根据请求量和最大的上下文长度，按实际情况调整。

## 3 模型量化

本部分内容主要介绍如何对模型进行量化。主要包括 KV Cache 量化和模型参数量化。总的来说，量化是一种以参数或计算中间结果精度下降换空间节省（以及同时带来的性能提升）的策略。

正式介绍 LMDeploy 量化方案前，需要先介绍两个概念：

- 计算密集（compute-bound）: 指推理过程中，绝大部分时间消耗在数值计算上；针对计算密集型场景，可以通过使用更快的硬件计算单元来提升计算速。
- 访存密集（memory-bound）: 指推理过程中，绝大部分时间消耗在数据读取上；针对访存密集型场景，一般通过减少访存次数、提高计算访存比或降低访存量来优化。

常见的 LLM 模型由于 Decoder Only 架构的特性，实际推理时大多数的时间都消耗在了逐 Token 生成阶段（Decoding 阶段），是典型的访存密集型场景。

那么，如何优化 LLM 模型推理中的访存密集问题呢？ 我们可以使用 **KV Cache 量化**和 **4bit Weight Only 量化（W4A16）**。KV Cache 量化是指将逐 Token（Decoding）生成过程中的上下文 K 和 V 中间结果进行 INT8 量化（计算时再反量化），以降低生成过程中的显存占用。4bit Weight 量化，将 FP16 的模型权重量化为 INT4，Kernel 计算时，访存量直接降为 FP16 模型的 1/4，大幅降低了访存成本。Weight Only 是指仅量化权重，数值计算依然采用 FP16（需要将 INT4 权重反量化）。

### 量化最佳实践

![0e71189cf478885eec20e406f3e7207b_quant drawio](https://github.com/superkong001/InternLM_Learning/assets/37318654/8240d119-1ef6-4b1e-a3ff-9456775344c3)

具体步骤如下。

- Step1：优先尝试正常（非量化）版本，评估效果。
    - 如果效果不行，需要尝试更大参数模型或者微调。
    - 如果效果可以，跳到下一步。
- Step2：尝试正常版本+KV Cache 量化，评估效果。
    - 如果效果不行，回到上一步。
    - 如果效果可以，跳到下一步。
- Step3：尝试量化版本，评估效果。
    - 如果效果不行，回到上一步。
    - 如果效果可以，跳到下一步。
- Step4：尝试量化版本+ KV Cache 量化，评估效果。
    - 如果效果不行，回到上一步。
    - 如果效果可以，使用方案。

另外需要补充说明的是，使用哪种量化版本、开启哪些功能，除了上述流程外，**还需要考虑框架、显卡的支持情况**，比如有些框架可能不支持 W4A16 的推理，那即便转换好了也用不了。

根据实践经验，一般情况下：

- 精度越高，显存占用越多，推理效率越低，但一般效果较好。
- Server 端推理一般用非量化版本或半精度、BF16、Int8 等精度的量化版本，比较少使用更低精度的量化版本。
- 端侧推理一般都使用量化版本，且大多是低精度的量化版本。这主要是因为计算资源所限。

以上是针对项目开发情况，如果是自己尝试（玩儿）的话：

- 如果资源足够（有GPU卡很重要），那就用非量化的正常版本。
- 如果没有 GPU 卡，只有 CPU（不管什么芯片），那还是尝试量化版本。
- 如果生成文本长度很长，显存不够，就开启 KV Cache。

# 实操记录

## 环境搭建

从clone环境下载

conda create -n CONDA_ENV_NAME --clone /share/conda_envs/internlm-base
或者：/root/share/install_conda_env_internlm_base.sh lmdeploy

安装 lmdeploy
```bash
# 解决 ModuleNotFoundError: No module named 'packaging' 问题
pip install packaging
# 使用 flash_attn 的预编译包解决安装过慢问题
pip install /root/share/wheels/flash_attn-2.4.2+cu118torch2.0cxx11abiTRUE-cp310-cp310-linux_x86_64.whl

pip install 'lmdeploy[all]==v0.1.0'
```

## 服务部署

离线转换 

```bash
# 转换模型（FastTransformer格式） 把 huggingface 格式的模型，转成 turbomind 推理格式，得到一个 workspace 目录
lmdeploy convert internlm-chat-7b  /root/share/temp/model_repos/internlm-chat-7b/
```

执行完成后将会在当前目录生成一个 workspace 的文件夹。这里面包含的就是 TurboMind 和 Triton “模型推理”需要到的文件。

<img width="566" alt="af18af74989dcb83648e7738f8e5ffe4_4" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/be62d3e5-2913-4f2d-827e-f8c593251cef">

weights 和 tokenizer 目录分别放的是拆分后的参数和 Tokenizer

每一份参数第一个 0 表示“层”的索引，后面的那个0表示 Tensor 并行的索引，因为我们只有一张卡，所以被拆分成 1 份。如果有两张卡可以用来推理，则会生成0和1两份，也就是说，会把同一个参数拆成两份。比如 layers.0.attention.w_qkv.0.weight 会变成 layers.0.attention.w_qkv.0.weight 和 layers.0.attention.w_qkv.1.weight。执行 lmdeploy convert 命令时，可以通过 --tp 指定（tp 表示 tensor parallel），该参数默认值为1（也就是一张卡）。

## TurboMind推理+API服务

首先，通过下面命令启动服务

```bash
# ApiServer+Turbomind   api_server => AsyncEngine => TurboMind
lmdeploy serve api_server ./workspace \
	--server_name 0.0.0.0 \
	--server_port 23333 \
	--instance_num 64 \
	--tp 1
```

上面的参数中 server_name 和 server_port 分别表示服务地址和端口，tp 参数我们之前已经提到过了，表示 Tensor 并行。还剩下一个 instance_num 参数，表示实例数，可以理解成 Batch 的大小。执行后如下图所示。

<img width="581" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/287f0b70-5a9a-4aa7-a05e-8a272f159c7f">

然后，我们可以新开一个窗口，执行下面的 Client 命令。如果使用官方机器，可以打开 vscode 的 Terminal，执行下面的命令。

```bash
# ChatApiClient+ApiServer（注意是http协议，需要加http）
lmdeploy serve api_client http://localhost:23333
```

<img width="711" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/ffeaf620-37ad-40e3-b80f-ff0708641d8c">

本机： ssh -CNg -L 23333:127.0.0.1:23333 root@ssh.intern-ai.org.cn -p <你的ssh端口号>

<img width="712" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/ca46b763-874b-4afc-b2e8-fa5e954eb650">

以 v1/chat/completions 接口为例，简单试一下。

```json
{
  "model": "internlm-chat-7b",
  "messages": "写一首春天的诗",
  "temperature": 0.7,
  "top_p": 1,
  "n": 1,
  "max_tokens": 512,
  "stop": false,
  "stream": false,
  "presence_penalty": 0,
  "frequency_penalty": 0,
  "user": "string",
  "repetition_penalty": 1,
  "renew_session": false,
  "ignore_eos": false
}
```

请求结果如下：

<img width="895" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/8ba9ec9f-b818-4f74-aad3-94e2f380a61b">

<img width="905" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/78707742-acd2-49e7-a7f0-cd32a0e8544a">

## 网页 Demo 演示

是将 Gradio 作为前端 Demo 演示。在上面基础上，不执行后面的 api_client 或 triton_client，而是执行 gradio。

TurboMind 服务作为后端:API Server 的启动后，直接启动作为前端的 Gradio

```bash
# Gradio+ApiServer。必须先开启 Server，此时 Gradio 为 Client
lmdeploy serve gradio http://0.0.0.0:23333 \
	--server_name 0.0.0.0 \
	--server_port 6006 \
	--restful_api True
 ```

由于 Gradio 需要本地访问展示界面，因此也需要通过 ssh 将数据转发到本地。命令如下：

ssh -CNg -L 6006:127.0.0.1:6006 root@ssh.intern-ai.org.cn -p <你的 ssh 端口号>

<img width="923" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/21baff9f-1fe1-490a-990f-dc088bee01ae">

# 模型量化
## KV Cach 量化（INT8）

转成 turbomind 推理格式后，已经得到一个 workspace 目录，就可以获取量化参数

主要思路是通过计算给定输入样本在每一层不同位置处计算结果的统计情况。

- 对于 Attention 的 K 和 V：取每个 Head 各自维度在所有Token的最大、最小和绝对值最大值。对每一层来说，上面三组值都是 `(num_heads, head_dim)` 的矩阵。这里的统计结果将用于本小节的 KV Cache。
- 对于模型每层的输入：取对应维度的最大、最小、均值、绝对值最大和绝对值均值。每一层每个位置的输入都有对应的统计值，它们大多是 `(hidden_dim, )` 的一维向量，当然在 FFN 层由于结构是先变宽后恢复，因此恢复的位置维度并不相同。这里的统计结果用于下个小节的模型参数量化，主要用在缩放环节（回顾PPT内容）。

### 计算 minmax

```bash
# 计算 minmax命令
lmdeploy lite calibrate \
  $HF_MODEL \
  --calib-dataset 'ptb' \            # 校准数据集，支持 c4, ptb, wikitext2, pileval
  --calib-samples 128 \              # 校准集的样本数，如果显存不够，可以适当调小
  --calib-seqlen 2048 \              # 单条的文本长度，如果显存不够，可以适当调小
  --work-dir $WORK_DIR \             # 保存 Pytorch 格式量化统计参数和量化后权重的文件夹
```

命令行中选择 128 条输入样本，每条样本长度为 2048，数据集选择 C4，输入模型后就会得到上面的各种统计值。

值得说明的是，如果显存不足，可以适当调小 samples 的数量或 sample 的长度。

### 获取量化参数

```bash
# 通过 minmax 获取量化参数,主要就是利用下面这个公式，获取每一层的 K V 中心值（zp）和缩放值（scale）。
zp = (min+max) / 2
scale = (max-min) / 255
quant: q = round( (f-zp) / scale)
dequant: f = q * scale + zp
```

```bash
# 通过 minmax 获取量化参数命令
lmdeploy lite kv_qparams \
  $WORK_DIR  \                                        # 上一步的结果
  workspace/triton_models/weights/ \                  # 保存量化参数的目录，推理要用
  --num-tp 1  \                                       # Tensor 并行使用的 GPU 数，和 deploy.py 保持一致
```

在这个命令中，`num_tp` 表示 Tensor 的并行数。每一层的中心值和缩放值会存储到 `workspace` 的参数目录中以便后续使用。`kv_sym` 为 `True` 时会使用另一种（对称）量化方法，它用到了第一步存储的绝对值最大值，而不是最大值和最小值。

kv_qparams 会在 weights 目录生成 fp32 缩放系数，文件格式是 numpy.tofile 产生的二进制。

也可以先把 turbomind_dir 设成私有目录，再把缩放系数拷贝进 workspace/triton_models/weights/

### 修改配置操作

修改 workspace/triton_models/weights/config.ini：quant_policy 设置为 4。表示打开 kv_cache int8

另外，如果用的是 TurboMind1.0，还需要修改参数 `use_context_fmha`，将其改为 0。

### 测试聊天效果

lmdeploy chat turbomind ./workspace

如对象为 internlm-chat-7b 模型。 测试方法：

1. 使用 deploy.py 转换模型，修改 workspace 配置中的最大并发数；调整 llama_config.ini 中的请求
2. 编译执行 bin/llama_triton_example，获取 fp16 版本在不同 batch_size 的显存情况
3. 开启量化，重新执行 bin/llama_triton_example，获取 int8 版本在不同 batch_size 显存情况

## W8A8 量化

需要安装了 lmdeploy 和 openai/triton组件：

```bash
pip install lmdeploy
pip install triton>=2.1.0
```

### 8bit 权重量化

如果需要进行 8 bit 权重模型推理，可以直接从 LMDeploy 的 model zoo 下载已经量化好的 8bit 权重模型。

以8bit 的 Internlm-chat-7B 模型为例，可以从 model zoo 直接下载：

```shell
git-lfs install
git clone https://huggingface.co/lmdeploy/internlm-chat-7b-w8 (coming soon)
```

进行 8bit 权重量化需要经历以下三步：

1. **权重平滑**：首先对语言模型的权重进行平滑处理，以便更好地进行量化。
2. **模块替换**：使用 `QRSMNorm` 和 `QLinear` 模块替换原模型 `DecoderLayer` 中的 `RSMNorm` 模块和 `nn.Linear` 模块。`lmdeploy/pytorch/models/q_modules.py` 文件中定义了这些量化模块。
3. **保存量化模型**：完成上述必要的替换后，我们即可保存新的量化模型。

在`lmdeploy/lite/api/smooth_quantity.py`脚本中已经实现了以上三个步骤。例如，可以通过以下命令得到量化后的 Internlm-chat-7B 模型的模型权重：

```shell
# 手动将原 16bit 权重量化为 8bit，并保存至 `internlm-chat-7b-w8` 目录下
lmdeploy lite smooth_quant internlm/internlm-chat-7b --work-dir ./internlm-chat-7b-w8
```

### W4A16 量化

W4A16中的A是指Activation，保持FP16，只对参数进行 4bit 量化。使用过程也是三步

量化权重模型。利用前面得到的统计值对参数进行量化，具体又包括两小步：

- 缩放参数。主要是性能上的考虑。
- 整体量化。

仅需执行一条命令，就可以完成模型量化工作。量化结束后，权重文件存放在 $WORK_DIR 下

```bash
# 量化权重模型
lmdeploy lite auto_awq \
   $HF_MODEL \                       # Model name or path, either model repo name on huggingface hub like 'internlm/internlm-chat-7b', or a model path in local host
  --calib-dataset 'ptb' \            # Calibration dataset, supports c4, ptb, wikitext2, pileval
  --calib-samples 128 \              # Number of samples in the calibration set, if memory is insufficient, you can appropriately reduce this
  --calib-seqlen 2048 \              # Length of a single piece of text, if memory is insufficient, you can appropriately reduce this
  --w-bits 4 \                       # Bit number for weight quantization
  --w-group-size 128 \               # Group size for weight quantization statistics
  --work-dir $WORK_DIR               # Folder storing Pytorch format quantization statistics parameters and post-quantization weight

# 样例
lmdeploy lite auto_awq \
  --model  /root/share/temp/model_repos/internlm-chat-7b/ \
  --w_bits 4 \
  --w_group_size 128 \
  --work_dir ./quant_output

# 可选参数不填写，可使用默认的
lmdeploy lite auto_awq internlm/ianternlm-chat-7b --work-dir internlm-chat-7b-4bit
```

命令中 `w_bits` 表示量化的位数，`w_group_size` 表示量化分组统计的尺寸，`work_dir` 是量化后模型输出的位置。这里需要特别说明的是，因为没有 `torch.int4`，所以实际存储时，8个 4bit 权重会被打包到一个 int32 值中。所以，如果你把这部分量化后的参数加载进来就会发现它们是 int32 类型的。

最后一步：转换成 TurboMind 格式。

```bash
# 转换模型的layout，存放在默认路径 ./workspace 下
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128
```

这个 `group-size` 就是上一步的那个 `w_group_size`。如果不想和之前的 `workspace` 重复，可以指定输出目录：`--dst_path`，比如：

```bash
lmdeploy convert  internlm-chat-7b ./quant_output \
    --model-format awq \
    --group-size 128 \
    --dst_path ./workspace_quant
```

### 测试聊天效果

lmdeploy chat torch ./internlm-chat-7b-w8

启动gradio服务

lmdeploy serve gradio ./internlm-chat-7b-4bit --server-name {ip_addr} --server-port {port}




