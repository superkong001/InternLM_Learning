1 环境配置

进入 conda 环境之后，使用以下命令从本地克隆一个已有的 pytorch 2.0.1 的环境

bash # 请每次使用 jupyter lab 打开终端时务必先执行 bash 命令进入 bash 中

bash /root/share/install_conda_env_internlm_base.sh internlm  # 执行该脚本文件来安装项目实验环境
<img width="533" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/5f08c2e8-fe20-4208-b896-6a2116069dac">

conda activate InternLM
并在环境中安装运行 demo 所需要的依赖。

# 升级pip
python -m pip install --upgrade pip

pip install modelscope==1.9.5

pip install transformers==4.35.2

pip install streamlit==1.24.0

pip install sentencepiece==0.1.99

pip install accelerate==0.24.1

1.2 模型下载

import torch

from modelscope import snapshot_download, AutoModel, AutoTokenizer

import os

model_dir = snapshot_download('Shanghai_AI_Laboratory/internlm-chat-7b', cache_dir='/root/data/model', revision='v1.0.3')

1.3 LangChain 相关环境配置
使用 huggingface 镜像下载

<img width="735" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/cc95693c-cf91-48b7-af99-18bd433698e5">

<img width="711" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/00a4502a-2519-4022-81ee-99b232164078">

1.4 下载 NLTK 相关资源

下载 nltk 资源并解压到服务器上

cd /root

git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages

cd nltk_data

mv packages/*  ./

cd tokenizers

unzip punkt.zip

cd ../taggers

unzip averaged_perceptron_tagger.zip

下载好慢，好慢！！！
<img width="467" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/7be845c3-99ce-4a17-9dd8-f121b06f8fc5">

1.5 下载本项目代码
<img width="461" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/d655179b-29bb-4c8b-bf0b-5509b5851797">

2 知识库搭建
2.1 数据收集
将远程开源仓库 Clone 到本地，可以使用以下命令：

# 进入到数据库盘
cd /root/data
# clone 上述开源仓库
git clone https://gitee.com/open-compass/opencompass.git
git clone https://gitee.com/InternLM/lmdeploy.git
git clone https://gitee.com/InternLM/xtuner.git
git clone https://gitee.com/InternLM/InternLM-XComposer.git
git clone https://gitee.com/InternLM/lagent.git
git clone https://gitee.com/InternLM/InternLM.git

<img width="478" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/b564bbdc-8566-4628-9a0d-874e38132e29">

<img width="420" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/2fbb1bb5-e76f-41df-a807-4381e0b10000">

<img width="735" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/a56e8ede-fb99-4d2d-8faa-6ae0bfe55be6">

3 InternLM 接入 LangChain

<img width="745" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/9f332879-eda8-4193-8360-af5e163dfa12">

4 构建检索问答链

<img width="766" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/79d63b41-efe8-4bca-9d11-f3d69f54f0f6">

运行提示的IP又不对。。。

<img width="541" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/882461f6-794b-4862-8f7e-dce5246a4ccf">

<img width="742" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/10fb7d3b-dbd7-4505-975a-c9035a4a58e5">

<img width="752" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/5568013b-53f9-4e0d-b4de-60ec9405773d">














