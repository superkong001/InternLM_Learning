# 理论
7B, 14亿参数，FP16半精度占2个字节，估算需要14G内存

<img width="601" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/99bb7947-1a50-4152-aae5-139459ce6000">

token by token，decode-only, 所以需要保存历史对话，那KV会多

<img width="592" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/0a8ae3cf-e4c3-45e9-92b8-939861ccd622">

<img width="671" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/442ed71f-f405-437b-a985-83b5ec95accf">


