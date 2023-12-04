# COIG-CQIA：Quality is All you need for Chinese Instruction Fine-tuning

\[ [English](README.md) | 中文 \]

欢迎来到COIG-CQIA，COIG-CQIA全称为**Chinese Open Instruction Generalist - Quality is All You Need**， 是一个开源的高质量指令微调数据集，旨在为中文NLP社区提供**高质量**且符合**人类交互行为**的指令微调数据。


## 项目简介

COIG-CQIA以中文互联网获取到的问答及文章作为原始数据，经过深度清洗、重构及人工审核构建而成。本项目受*LIMA: Less Is More for Alignment*等研究启发，使用少量高质量的数据即可让大语言模型学习到人类交互行为，因此在数据构建中我们十分注重数据的来源、质量与多样性，数据集详情请见[数据](#数据详情)及接下来的论文。

## 更新

- **[2023.12.04]** 🎉发布v0.1版数据，基于Yi-6B-base和Yi-34B-base使用v0.1版数据全量微调的SFT模型

## Models

利用COIG-CQIA数据，我们开发了一系列基于Yi的SFT模型，更多版本即将发布。

| 模型名称 | 基座模型 | 下载链接 |
| --- | --- | --- |
| CQIA-Yi-6B-v0.1 | Yi-6B-base | [Download](#) |
| CQIA-Yi-34B-v0.1 | Yi-34B-base | [Download](#) |

### 使用方法

```python
from transformers import AutoModel
```

## 效果展示

<details>
<summary><b>Logical Reasoning</b></summary>
<p>

> Input: 

> Response: 

</p>
</details>

## 数据详情

### 数据格式

```json
{
    "instruction": "示例问题或者指令。",
    "input": "示例问题或指令的补充。",
    "output": "对输入的回复。",
    "task_type": {
        "major": ["问答"],
        "minor": ["百科问答"]
    },
    "domain": ["百科", "医疗"],
    "answer_from": "human",
    "human_verified": true,
    "copyright": "作者及版权信息。",
}
```

### 数据字段

- `instruction`: 用于输入的指令或者问题。
- `input`: 问题或指令的补充内容。
- `output`: 输入对应的回答。
- `task_type`: 表示该数据所属的主要任务类型和细分任务类型。
- `domain`: 该数据所属领域。
- `answer_from`: 回答是人类撰写的还是大模型撰写的，本数据集中绝大部分是由人类撰写的回答，少部分由大模型生成（经过了人工验证）。
- `human_verified`: 该数据是否又人类核验过。
- `copyright`: 包括该数据的版权信息，包括作者等。

### 数据详情

<details>
<summary><b>社交媒体&论坛</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| 知乎        | 8837      | [[网址链接]](https://www.zhihu.com/) | 经过过阶段的数据质量筛选和人工验证。 |
| 豆瓣       | 3132    | [[网址链接]](https://www.douban.com/) | 人工撰写多样的prompt模板构造而成。 |
| 小红书       | 1508    | [[网址链接]](https://www.xiaohongshu.com/explore) | 人工撰写多样的prompt模板构造而成。 |
| Segmentfault       | 458    | [[网址链接]](https://segmentfault.com/) | 规则方式清洗与筛选，并经过人工验证。 |
| **总量**         | **13935** | -      | -                                       |

</details>

<details>
<summary><b>通用百科</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| 百科文章        | 980      | 从网络中收集。[[网址链接]](https://10why.net/) [[网址链接]](https://www.eetree.cn/wiki/eebaike) [[网址链接]](https://www.nongyie.com/) [[网址链接]](https://www.gkket.com/gkwk/) | 规则方式清洗与筛选，并经过人工验证。 |
| 中国大百科全书       | 1706    | [[网址链接]](https://www.zgbk.com/) | 人工撰写多样的prompt模板构造而成。 |
| wikiHow中文       | 1876    | [[网址链接]](https://zh.wikihow.com/首页)&[[公开数据集]](https://github.com/esbatmop/MNBVC/tree/main) | 规则方式清洗与筛选。 |
| **总量**         | **4571** | -      | -                                       |

</details>

<details>
<summary><b>考试&试题</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| 高考&中考        | 2000      | [[公开数据集]](https://huggingface.co/datasets/BAAI/COIG) | - |
| 研究生入学考试       | 475    | 从网络中收集 | 规则方式清洗与筛选。 |
| 逻辑推理题       | 422    | 从网络中收集 | 规则方式清洗与筛选。 |
| **总量**         | **2897** | -      | -                                       |

</details>

<details>
<summary><b>人类价值观</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| 100poison         | 906      | [[公开数据集]](https://modelscope.cn/datasets/damo/100PoisonMpts/summary) | - |
| COIG-human-value  | 101      | [[公开数据集]](https://huggingface.co/datasets/BAAI/COIG) | 经人工审核数据质量 |
| **总量**         | **1007** | -      | -                                       |

</details>

<details>
<summary><b>中国传统文化</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| 中华传统文化试题         | 232      | 从网络中收集 | 规则方式清洗与筛选，并经过人工验证。 |
| 成语释义  | 112      | [[公开数据集]](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | 规则方式清洗与筛选，并经过人工验证。 |
| 古诗词撰写  | 47      | [[公开数据集]](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | 规则方式清洗与筛选，并经过人工验证。 |
| 文言文互译  | 112      | [[公开数据集]](https://huggingface.co/datasets/YeungNLP/firefly-train-1.1M) | 规则方式清洗与筛选，并经过人工验证。 |
| **总量**         | **1112** | -      | -                                       |

</details>

<details>
<summary><b>金融&经管领域</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| MBA百科       | 10689    | [[网址链接]](https://wiki.mbalib.com/wiki/首页) | 人工撰写多样的prompt模板构造而成。 |
| 金融NLP任务  | 600      | [[公开数据集]](https://huggingface.co/datasets/BAAI/COIG-PC) | 人工核验数据质量 |
| **总量**         | **12689** | -      | -                                       |

</details>

<details>
<summary><b>医疗领域</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| 医疗百科       | 8351    | [[网址链接]](www.baikemy.com) | 人工撰写多样的prompt模板构造而成。 |
| 医疗文章  | 186      | [[网址链接]](https://51zyzy.com/article/list.html) [[网址链接]](https://baobao.baidu.com/dailyjnl/list/13.html) | 规则方式清洗与筛选。 |
| **总量**         | **8537** | -      | -                                       |

</details>

<details>
<summary><b>法律领域</b></summary>

| 类别          | 数量 | 来源 | 构造方式                     |
| ----------------- | -------- | ------ | --------------------------------------- |
| 法律研究生入学考试       | 2645    | 从网络中收集 | 规则方式清洗与筛选。 |
| **总量**         | **2645** | -      | -                                       |

</details>

## Citation

如果本项目为您的研究带来了帮助，请参考以下格式引用：

```bibtex
@misc{COIG-CQIA,
  author = {},
  title = {COIG-CQIA: Quality is All you need for Chinese Instruction Fine-tuning},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/paralym/COIG-CQIA}},
}
```

本数据集中也包含了以下公开数据：
```bibtex
@article{zhang2023chinese,
  title={Chinese open instruction generalist: A preliminary release},
  author={Zhang, Ge and Shi, Yemin and Liu, Ruibo and Yuan, Ruibin and Li, Yizhi and Dong, Siwei and Shu, Yu and Li, Zhaoqun and Wang, Zekun and Lin, Chenghua and others},
  journal={arXiv preprint arXiv:2304.07987},
  year={2023}
}
@misc{Firefly,
  author = {Jianxin Yang},
  title = {Firefly(流萤): 中文对话式大语言模型},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yangjianxin1/Firefly}},
}
@misc{xu2023cvalues,
    title={CValues: Measuring the Values of Chinese Large Language Models from Safety to Responsibility}, 
    author={Guohai Xu and Jiayi Liu and Ming Yan and Haotian Xu and Jinghui Si and Zhuoran Zhou and Peng Yi and Xing Gao and Jitao Sang and Rong Zhang and Ji Zhang and Chao Peng and Fei Huang and Jingren Zhou},
    year={2023},
    eprint={2307.09705},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
  }
```
