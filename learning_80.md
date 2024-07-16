# SSH连接与端口映射

```bash
> ssh -p 34531 root@ssh.intern-ai.org.cn -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
```

<img width="695" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/69985cf9-59aa-4d6d-971f-5408917c9c0b">

<img width="811" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/d133c2eb-6339-449a-bd16-868b005c3e4d">

ssh -p 34531 root@ssh.intern-ai.org.cn -CNg -L 7860:127.0.0.1:7860 -o StrictHostKeyChecking=no

## 运行hello_world.py

<img width="653" alt="image" src="https://github.com/superkong001/InternLM_Learning/assets/37318654/ac928260-54a8-49f5-a0d7-577aa43292b6">

## wordcound

```python
import re

def word_count(text):
    print("text is: \n",text)

    # 将所有大写字母转换为小写，并使用正则表达式去除特殊空白字符
    text = re.sub(r'\s+', ' ', text.lower().strip())
    
    # 按空格分词
    words = text.split(' ')
    # 初始化字典来存储单词计数
    word_dict = {}
    
    # 循环统计每个单词的出现次数
    for word in words:
        if word in word_dict:
            word_dict[word] += 1
        else:
            word_dict[word] = 1
    
    # 按照出现次数从高到低排序
    sorted_words = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)

    print(sorted_words)
    return sorted_words

text = """
Hello world!  
This is an example.  
Word count is fun.  
Is it fun to count words?  
Yes, it is fun!
"""

word_count(text)
```

<img width="766" alt="image" src="https://github.com/user-attachments/assets/62af2646-ae71-4c72-82f7-624de805e4d6">


