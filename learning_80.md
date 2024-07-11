# SSH连接与端口映射

```bash
> ssh -p 34531 root@ssh.intern-ai.org.cn -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
```

<img width="695" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/69985cf9-59aa-4d6d-971f-5408917c9c0b">

<img width="811" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/d133c2eb-6339-449a-bd16-868b005c3e4d">

ssh -p 34531 root@ssh.intern-ai.org.cn -CNg -L 7860:127.0.0.1:7860 -o StrictHostKeyChecking=no

## 运行hello_world.py

<img width="653" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/ac928260-54a8-49f5-a0d7-577aa43292b6">



