# ChatGLM-6B-Slim

## 介绍
ChatGLM-6B-Slim是在ChatGLM-6B的基础上通过裁剪词表构建的。因为ChatGLM-6B使用了icetk，在其词表中，前20000个token是预留给图片的，在文本模型中没有用到这些图片token，但是在infer和微调的时候，这些token对应的embedding依然需要被加载，并且在解码每一个token的时候需要多计算20K个logits，会占用不少显存。因此将这一部分token裁剪掉以节省显存和计算。

除了词表外，ChatGLM-6B-Slim的其他结构与ChatGLM-6B完全一致，性能也完全一样，可以认为是ChatGLM-6B的一个低显存版等价平替。其使用方式和ChatGLM-6B完全一致，只需要修改加载模型和tokenizer部分的代码。

不过，由于 ChatGLM-6B 的规模较小，目前已知其具有相当多的[**局限性**](#局限性)，如事实性/数学逻辑错误，可能生成有害/有偏见内容，较弱的上下文能力，自我认知混乱，以及对英文指示生成与中文指示完全矛盾的内容。请大家在使用前了解这些问题，以免产生误解。更大的基于1300亿参数[GLM-130B](https://github.com/THUDM/GLM-130B)的ChatGLM正在内测开发中。

*Read this in [English](README_en.md).*
关于 ChatGLM-6B 的更多详情请参考官方[repo](https://github.com/THUDM/ChatGLM-6B)

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
完整的模型实现可以在 [Hugging Face Hub](https://huggingface.co/silver/chatglm-6b-slim) 上查看。如果你从 Hugging Face Hub 上下载checkpoint的速度较慢，也可以从[这里](https://cloud.tsinghua.edu.cn/d/fb9f16d6dc8f482596c2/)手动下载。
