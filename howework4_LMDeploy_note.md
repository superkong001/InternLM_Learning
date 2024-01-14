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

# 实践

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

