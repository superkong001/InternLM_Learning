<img width="361" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/50b96e91-3359-40ca-a010-70fe45f69d2d">

RAG

<img width="293" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/de5ba156-4de9-4e4f-9418-fb854ce2ff4d">

1.先对文本进行分块：LangChain 提供了多种文本分块工具，此处我们使用字符串递归分割器，并选择分块大小为 500，块重叠长度为 150（由于篇幅限制，此处没有展示切割效果，学习者可以自行尝试一下，想要深入学习 LangChain 文本分块可以参考教程 《LangChain - Chat With Your Data》

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=150)
split_docs = text_splitter.split_documents(docs)
2.接着对文本块进行向量化：开源词向量模型 Sentence Transformer 来进行文本向量化
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="/root/data/model/sentence-transformer")

3.选择 Chroma 作为向量数据库，基于上文分块后的文档以及加载的开源向量化模型，将语料加载到指定路径下的向量数据库：

from langchain.vectorstores import Chroma

# 定义持久化路径
persist_directory = 'data_base/vector_db/chroma'
# 加载数据库
vectordb = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings,
    persist_directory=persist_directory  # 允许我们将persist_directory目录保存到磁盘上
)
# 将加载的向量数据库持久化到磁盘上
vectordb.persist()
<img width="449" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/0e45ddac-d4da-4d52-9cf3-4e84f4de902a">

