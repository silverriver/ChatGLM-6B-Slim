# ChatGLM-6B

## 介绍
ChatGLM-6B-Slim是在ChatGLM-6B的基础上通过裁剪词表构建的。因为ChatGLM-6B使用了icetk，在其词表中，前20000个token是预留给图片的，在文本模型中没有用到这些图片token，但是在infer和微调的时候，这些token对应的embedding依然需要被加载，并且在解码每一个token的时候需要多计算20K个logits，会占用不少显存。因此将这一部分token裁剪掉以节省显存和计算。

ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 [General Language Model (GLM)](https://github.com/THUDM/GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。
ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答。更多信息请参考我们的[博客](https://chatglm.cn/blog)。

不过，由于 ChatGLM-6B 的规模较小，目前已知其具有相当多的[**局限性**](#局限性)，如事实性/数学逻辑错误，可能生成有害/有偏见内容，较弱的上下文能力，自我认知混乱，以及对英文指示生成与中文指示完全矛盾的内容。请大家在使用前了解这些问题，以免产生误解。更大的基于 1300 亿参数 [GLM-130B](https://github.com/THUDM/GLM-130B) 的 ChatGLM 正在内测开发中。

*Read this in [English](README_en.md).*

## 更新信息
**[2023/03/23]** 增加API部署（感谢 [@LemonQu-GIT](https://github.com/LemonQu-GIT)）。增加Embedding量化模型[ChatGLM-6B-INT4-QE](https://huggingface.co/THUDM/chatglm-6b-int4-qe)。增加对基于Apple Silicon的Mac上GPU加速的支持。

**[2023/03/19]** 增加流式输出接口 `stream_chat`，已更新到网页版和命令行 Demo。修复输出中的中文标点。增加量化后的模型 [ChatGLM-6B-INT4](https://huggingface.co/THUDM/chatglm-6b-int4)

## 友情链接
以下是部分基于本仓库开发的开源项目：
* [ChatGLM-MNN](https://github.com/wangzhaode/ChatGLM-MNN): 一个基于 MNN 的 ChatGLM-6B C++ 推理实现，支持根据显存大小自动分配计算任务给 GPU 和 CPU
* [ChatGLM-Tuning](https://github.com/mymusise/ChatGLM-Tuning): 基于 LoRA 对 ChatGLM-6B 进行微调。类似的项目还包括 [Humanable ChatGLM/GPT Fine-tuning | ChatGLM 微调](https://github.com/hscspring/hcgf)
* [langchain-ChatGLM](https://github.com/imClumsyPanda/langchain-ChatGLM)：基于本地知识的 ChatGLM 应用，基于LangChain
* [bibliothecarius](https://github.com/coderabbit214/bibliothecarius)：快速构建服务以集成您的本地数据和AI模型，支持ChatGLM等本地化模型接入。
* [闻达](https://github.com/l15y/wenda)：大型语言模型调用平台，基于 ChatGLM-6B 实现了类 ChatPDF 功能

以下是部分针对本项目的教程/文档：
* [Windows部署文档](https://github.com/ZhangErling/ChatGLM-6B/blob/main/deployment_windows.md)

如果你有其他好的项目/教程的话，欢迎参照上述格式添加到README中并提出 [PR](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request-from-a-fork).

## 使用方式

### 硬件需求

| **量化等级**    | **最低 GPU 显存** |
| -------------- | ----------------- |
| FP16（无量化）   | 13 GB             |
| INT8           | 10 GB              |
| INT4           | 6 GB               |

### 环境安装

使用 pip 安装依赖：`pip install -r requirements.txt`，其中 `transformers` 库版本推荐为 `4.26.1`，但理论上不低于 `4.23.1` 即可。

### 代码调用 

可以通过如下代码调用 ChatGLM-6B-Slim 模型来生成对话：

```python
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("silver/chatglm-6b-slim", trust_remote_code=True)
>>> model = AutoModel.from_pretrained("silver/chatglm-6b-slim", trust_remote_code=True).half().cuda()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。
>>> response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
>>> print(response)
晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:

1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。
2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。
3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。
4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。
5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。
6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。

如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。
```
完整的模型实现可以在 [Hugging Face Hub](https://huggingface.co/THUDM/chatglm-6b) 上查看。如果你从 Hugging Face Hub 上下载checkpoint的速度较慢，也可以从[这里](https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/)手动下载。
