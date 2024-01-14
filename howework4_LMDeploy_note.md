# 理论
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



