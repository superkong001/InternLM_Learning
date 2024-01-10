使用Lagent 工具调用 Demo 创作部署中出了一个问题：
由于前一个做网页形式写个小故事后，直接ctrl+C，在部署Lagent 工具后，再提交streamlit run /root/code/lagent/examples/react_web_demo.py --server.address 127.0.0.1 --server.port 6006时候报错，端口被占用，点了释放还是不行
解决方案：只能用重启大法，重启VSCODE，重启后恢复了

浦语·灵笔图文理解创作 Demo时有遇到一个问题：
运行以下命令后，浏览器输入http://127.0.0.1:6006，不能打开：
cd /root/code/InternLM-XComposer
python examples/web_demo.py  \
    --folder /root/model/Shanghai_AI_Laboratory/internlm-xcomposer-7b \
    --num_gpus 1 \
    --port 6006
解决方案：
看了一下web_demo.py代码，然后链接改成http://0.0.0.0:6006 能正常访问

另外，在模型下载和加载时：
由于本地是Windows 11 + AMD显卡，显卡不支持cuda和ROCm，导致加载7b模型，只能用float32
修改如下：
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True).to("cpu")
虽然跑成功了，但结果是原来13.6G的内存/显存空间，转成了float32后直接翻倍了。。。。

