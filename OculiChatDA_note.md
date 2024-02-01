## 准备微调数据集

### 准备lora数据集

```Bash
# 获取openxlab数据集
pip install openxlab #安装
pip install -U openxlab #版本升级
openxlab login #进行登录，输入对应的AK/SK
openxlab dataset info --dataset-repo OpenDataLab/MedFMC #数据集信息查看
openxlab dataset ls --dataset-repo OpenDataLab/MedFMC #数据集文件列表查看
openxlab dataset get --dataset-repo OpenDataLab/MedFMC #数据集下载
openxlab dataset download --dataset-repo OpenDataLab/MedFMC --source-path /README.md --target-path /path/to/local/folder #数据集文件下载
```

> https://huggingface.co/datasets/clip-benchmark/wds_vtab-diabetic_retinopathy/tree/main
> https://huggingface.co/datasets/Rami/Diabetic_Retinopathy_Preprocessed_Dataset_256x256/tree/main

### 准备Qlora数据集
> https://huggingface.co/datasets/Toyokolabs/retinoblastoma/tree/main


## 安装

```Bash
# 如果你是在 InternStudio 平台，则从本地 clone 一个已有 pytorch 2.0.1 的环境：
/root/share/install_conda_env_internlm_base.sh Oculi
#有问题删除：（光使用这个命令没用conda remove --name Oculi --all， 还要删除/root/.conda/envs/Oculi的所有文件）
conda activate Oculi
```

```Bash
mkdir xtuner && cd xtuner

# git clone -b v0.1.9  https://github.com/InternLM/xtuner
# 无法访问github的用户请从 gitee 拉取:
git clone https://gitee.com/Internlm/xtuner

# 进入源码目录
cd xtuner
# 从源码安装 XTuner
pip install -e '.[all]'

# 创建一个微调 oasst1 数据集的工作路径，进入
mkdir ~/ft-Oculi && cd ~/ft-Oculi
```

## 微调

### 准备配置文件

```Bash
# 列出所有内置配置
xtuner list-cfg
假如显示bash: xtuner: command not found的话可以考虑在终端输入 export PATH=$PATH:'/root/.local/bin'
==========================CONFIGS===========================
internlm2_20b_qlora_alpaca_e3
internlm2_20b_qlora_arxiv_gentitle_e3
internlm2_20b_qlora_code_alpaca_e3
internlm2_20b_qlora_colorist_e5
internlm2_20b_qlora_lawyer_e3
internlm2_20b_qlora_msagent_react_e3_gpu8
internlm2_20b_qlora_oasst1_512_e3
internlm2_20b_qlora_oasst1_e3
internlm2_20b_qlora_sql_e3
internlm2_7b_qlora_alpaca_e3
internlm2_7b_qlora_arxiv_gentitle_e3
internlm2_7b_qlora_code_alpaca_e3
internlm2_7b_qlora_colorist_e5
internlm2_7b_qlora_json_e3
internlm2_7b_qlora_lawyer_e3
internlm2_7b_qlora_msagent_react_e3_gpu8
internlm2_7b_qlora_oasst1_512_e3
internlm2_7b_qlora_oasst1_e3
internlm2_7b_qlora_sql_e3
internlm2_7b_w_tokenized_dataset
internlm2_7b_w_untokenized_dataset
internlm2_chat_20b_qlora_alpaca_e3
internlm2_chat_20b_qlora_code_alpaca_e3
internlm2_chat_20b_qlora_lawyer_e3
internlm2_chat_20b_qlora_oasst1_512_e3
internlm2_chat_20b_qlora_oasst1_e3
internlm2_chat_7b_qlora_alpaca_e3
internlm2_chat_7b_qlora_code_alpaca_e3
internlm2_chat_7b_qlora_lawyer_e3
internlm2_chat_7b_qlora_oasst1_512_e3
internlm2_chat_7b_qlora_oasst1_e3
internlm_20b_qlora_alpaca_e3
internlm_20b_qlora_alpaca_enzh_e3
internlm_20b_qlora_alpaca_enzh_oasst1_e3
internlm_20b_qlora_alpaca_zh_e3
internlm_20b_qlora_arxiv_gentitle_e3
internlm_20b_qlora_code_alpaca_e3
internlm_20b_qlora_colorist_e5
internlm_20b_qlora_lawyer_e3
internlm_20b_qlora_msagent_react_e3_gpu8
internlm_20b_qlora_oasst1_512_e3
internlm_20b_qlora_oasst1_e3
internlm_20b_qlora_open_platypus_e3
internlm_20b_qlora_sql_e3
internlm_7b_full_alpaca_e3
internlm_7b_full_alpaca_enzh_e3
internlm_7b_full_alpaca_enzh_oasst1_e3
internlm_7b_full_alpaca_zh_e3
internlm_7b_full_intern_repo_dataset_template
internlm_7b_full_oasst1_e3
internlm_7b_qlora_alpaca_e3
internlm_7b_qlora_alpaca_enzh_e3
internlm_7b_qlora_alpaca_enzh_oasst1_e3
internlm_7b_qlora_alpaca_zh_e3
internlm_7b_qlora_arxiv_gentitle_e3
internlm_7b_qlora_code_alpaca_e3
internlm_7b_qlora_colorist_e5
internlm_7b_qlora_json_e3
internlm_7b_qlora_lawyer_e3
internlm_7b_qlora_medical_e1
internlm_7b_qlora_moss_sft_all_e1
internlm_7b_qlora_moss_sft_all_e2_gpu8
internlm_7b_qlora_moss_sft_plugins_e1
internlm_7b_qlora_msagent_react_e3_gpu8
internlm_7b_qlora_oasst1_512_e3
internlm_7b_qlora_oasst1_e3
internlm_7b_qlora_oasst1_e3_hf
internlm_7b_qlora_oasst1_mmlu_e3
internlm_7b_qlora_open_platypus_e3
internlm_7b_qlora_openorca_e1
internlm_7b_qlora_sql_e3
internlm_7b_qlora_tiny_codes_e1
internlm_chat_20b_qlora_alpaca_e3
internlm_chat_20b_qlora_alpaca_enzh_e3
internlm_chat_20b_qlora_alpaca_enzh_oasst1_e3
internlm_chat_20b_qlora_alpaca_zh_e3
internlm_chat_20b_qlora_code_alpaca_e3
internlm_chat_20b_qlora_lawyer_e3
internlm_chat_20b_qlora_oasst1_512_e3
internlm_chat_20b_qlora_oasst1_e3
internlm_chat_20b_qlora_open_platypus_e3
internlm_chat_7b_qlora_alpaca_e3
internlm_chat_7b_qlora_alpaca_enzh_e3
internlm_chat_7b_qlora_alpaca_enzh_oasst1_e3
internlm_chat_7b_qlora_alpaca_zh_e3
internlm_chat_7b_qlora_arxiv_gentitle_e3
internlm_chat_7b_qlora_code_alpaca_e3
internlm_chat_7b_qlora_colorist_e5
internlm_chat_7b_qlora_lawyer_e3
internlm_chat_7b_qlora_medical_e1
internlm_chat_7b_qlora_oasst1_512_e3
internlm_chat_7b_qlora_oasst1_e3
internlm_chat_7b_qlora_open_platypus_e3
internlm_chat_7b_qlora_openorca_e1
internlm_chat_7b_qlora_sql_e3
internlm_chat_7b_qlora_tiny_codes_e1
=============================================================
```

```Bash
cd ~/ft-Oculi
xtuner copy-cfg internlm2_chat_7b_qlora_oasst1_e3 .
```

### 模型下载

从 OpenXLab 下载模型到本地

```Bash
# 创建一个目录，放模型文件，防止散落一地
mkdir ~/ft-Oculi/internlm2_chat_7b

# 装一下拉取模型文件要用的库
pip install modelscope

# 从 modelscope 下载下载模型文件
cd ~/ft-Oculi
apt install git git-lfs -y
git lfs install
git lfs clone https://modelscope.cn/Shanghai_AI_Laboratory/internlm2_chat_7b.git
```












