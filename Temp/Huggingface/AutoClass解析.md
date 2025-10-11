---
date: 2025-10-10
author:
  - Siyuan Liu
tags:
  - Huggingface
---
# AutoModel
```
from transformers import AutoModel

model = AutoModel.from_pretrained("google-bert/bert-base-cased")
```
transformer库中已经预先注册了一些经典模型的记录，这些记录就是一个`AutoModel`，调用`from_pretrained`就会自动根据模型名称去拉取服务器上这个模型的`config file`和`model file`，然后自动实例化模型

---
## AutoModelForSequenceClassification
对于文本（或序列）分类，你应该加载`AutoModelForSequenceClassification`

```
from transformers import AutoModelForSequenceClassification

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
pt_model = AutoModelForSequenceClassification.from_pretrained(model_name)
```

---
## 如果想将自定义的模型数据，注册进AutoModel中
```
from transformers import AutoConfig, AutoModel

AutoConfig.register("new-model", NewModelConfig)
AutoModel.register(NewModelConfig, NewModel)
```
首先需要给自定义的模型创建一个配置文件，比如`"new-model"`（详细该如何编写可以看后面），将这个自定义的模型的配置文件类命名为`NewModelConfig`。然后将`NewModelConfig`传递给`.register()`方法，命名为`NewModel`。之后就可以通过`AutoModel.from_pretrained("NewModel")`实例化模型了

---
# .from_pretrained()
## 参数
- **pretrained_model_name_or_path** (`str` or `os.PathLike`) — Can be either:
    
    - A string, the _model id_ of a pretrained model configuration hosted inside a model repo on huggingface.co.
    - A path to a _directory_ containing a configuration file saved using the [save_pretrained()](https://huggingface.co/docs/transformers/main/en/main_classes/configuration#transformers.PreTrainedConfig.save_pretrained) method, or the [save_pretrained()](https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.save_pretrained) method, e.g., `./my_model_directory/`.
    - A path or url to a saved configuration JSON _file_, e.g., `./my_model_directory/configuration.json`.
    
- [](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig.from_pretrained.cache_dir)**cache_dir** (`str` or `os.PathLike`, _optional_) — Path to a directory in which a downloaded pretrained model configuration should be cached if the standard cache should not be used.
- [](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig.from_pretrained.force_download)**force_download** (`bool`, _optional_, defaults to `False`) — Whether or not to force the (re-)download the model weights and configuration files and override the cached versions if they exist.
- [](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig.from_pretrained.proxies)**proxies** (`dict[str, str]`, _optional_) — A dictionary of proxy servers to use by protocol or endpoint, e.g., `{'http': 'foo.bar:3128', 'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
- [](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig.from_pretrained.revision)**revision** (`str`, _optional_, defaults to `"main"`) — The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any identifier allowed by git.
- [](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig.from_pretrained.return_unused_kwargs)**return_unused_kwargs** (`bool`, _optional_, defaults to `False`) — If `False`, then this function returns just the final configuration object.
    
    If `True`, then this functions returns a `Tuple(config, unused_kwargs)` where _unused_kwargs_ is a dictionary consisting of the key/value pairs whose keys are not configuration attributes: i.e., the part of `kwargs` which has not been used to update `config` and is otherwise ignored.
    
- [](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig.from_pretrained.trust_remote_code)**trust_remote_code** (`bool`, _optional_, defaults to `False`) — Whether or not to allow for custom models defined on the Hub in their own modeling files. This option should only be set to `True` for repositories you trust and in which you have read the code, as it will execute code present on the Hub on your local machine.
- [](https://huggingface.co/docs/transformers/main/en/model_doc/auto#transformers.AutoConfig.from_pretrained.kwargs\(additional)**kwargs(additional** keyword arguments, _optional_) — The values in kwargs of any keys which are configuration attributes will be used to override the loaded values. Behavior concerning key/value pairs whose keys are _not_ configuration attributes is controlled by the `return_unused_kwargs` keyword parameter.

---
## 运行流程

![[Pasted image 20251010204149.png]]
1. 去拉取服务器上这个模型的`config file`和`model file`
2. 打开`config file`根据模型 名字/地址 寻找应该使用的`AutoConfig`(例如GPT2，Deepseek-v1，etc.)
3. 实例化`AutoConfig`类。找到对应的`model class`
4. 根据`AutoConfig`和`model class`，实例化一个完整的`model`(还是随机的weits)
5. 从`model file`中加载权重到`model`中

---
# AutoConfig
## 示例
```
class BertConfig(PreTrainedConfig):
model_type = "bert"  
  
    def __init__(  
        self,  
        vocab_size=30522,  
        hidden_size=768,  
        num_hidden_layers=12,  
        num_attention_heads=12,  
        intermediate_size=3072,  
        hidden_act="gelu",  
        hidden_dropout_prob=0.1,  
        attention_probs_dropout_prob=0.1,  
        max_position_embeddings=512,  
        type_vocab_size=2,  
        initializer_range=0.02,  
        layer_norm_eps=1e-12,  
        pad_token_id=0,  
        use_cache=True,  
        classifier_dropout=None,  
        **kwargs,  
    ):  
        super().__init__(pad_token_id=pad_token_id, **kwargs)  
  
        self.vocab_size = vocab_size  
        self.hidden_size = hidden_size  
        self.num_hidden_layers = num_hidden_layers  
        self.num_attention_heads = num_attention_heads  
        self.hidden_act = hidden_act  
        self.intermediate_size = intermediate_size  
        self.hidden_dropout_prob = hidden_dropout_prob  
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  
        self.max_position_embeddings = max_position_embeddings  
        self.type_vocab_size = type_vocab_size  
        self.initializer_range = initializer_range  
        self.layer_norm_eps = layer_norm_eps  
        self.use_cache = use_cache  
        self.classifier_dropout = classifier_dropout  
  
  
class BertOnnxConfig(OnnxConfig):  
    @property  
    def inputs(self) -> Mapping[str, Mapping[int, str]]:  
        if self.task == "multiple-choice":  
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}  
        else:  
            dynamic_axis = {0: "batch", 1: "sequence"}  
        return OrderedDict(  
            [  
                ("input_ids", dynamic_axis),  
                ("attention_mask", dynamic_axis),  
                ("token_type_ids", dynamic_axis),  
            ]  
        )
```

创建指定`model`的`AutoConfig`类
```
from transformers import BertConfig

bert_config = BertConfig.from_pretrained("bert-base-cased")
```
## 自定义模型构建
你可以修改模型的配置类来改变模型的构建方式。配置指明了模型的属性，比如隐藏层或者注意力头的数量。

```
from transformers import AutoConfig

my_config = AutoConfig.from_pretrained("distilbert/distilbert-base-uncased", n_heads=12)
```

---
# 保存/加载 当前模型权重
```
from transformers import BertModel, BertConfig

bert_config = BertConfig.from_pretrained("google-bert/bert-base-cased")
bert_model = BertModel(bert_config)
.
.
.

bert_model.save_pretrained("my_bert_model") # 在 my_bert_model 目录下保存
```

```
from transformers import BertModel

bert_model = BertModel.from_pretrained("my_bert_model") 
```

---
# AutoTokenizer
分词器负责预处理文本，将文本转换为用于输入模型的数字数组。有多个用来管理分词过程的规则，包括如何拆分单词和在什么样的级别上拆分单词（在 [分词器总结](https://huggingface.co/docs/transformers/zh/tokenizer_summary) 学习更多关于分词的信息）。要记住最重要的是实例化的分词器名称要与模型的名称相同, 来确保和模型训练时使用相同的分词规则。

实例化`AutoTokenizer`
```
from transformers import AutoTokenizer

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

**将文本传入分词器**
```
encoding = tokenizer("We are very happy to show you the 🤗 Transformers library.")
print(encoding)
```
分词器返回了含有如下内容的字典:
- [input_ids](https://huggingface.co/docs/transformers/zh/glossary#input-ids)：用数字表示的 token
- [attention_mask](https://huggingface.co/docs/transformers/zh/.glossary#attention-mask)：应该关注哪些 token 的指示

**分词器也可以接受列表作为输入，并填充和截断文本，返回具有统一长度的批次**
```
pt_batch = tokenizer(
    ["We are very happy to show you the 🤗 Transformers library.", "We hope you don't hate it."],
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt",
)
```

# Trainer 优化训练循环
这个类只适用于huggingface的`PreTrainedModel` 和 `torch.nn.Module`

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, Trainer
from datasets import load_dataset

model = AutoModelForSequenceClassification.from_pretrained("distilbert/distilbert-base-uncased")

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")

dataset = load_dataset("rotten_tomatoes")  # doctest: +IGNORE_RESULT

# 创建一个给数据集分词的函数，并且使用 `map` 应用到整个数据集
def tokenize_dataset(dataset):
# 只对"text"列的内容进行分片
    return tokenizer(dataset["text"])

# batched=True 表示按批处理（默认批大小约 1000，可用 batch_size=… 调整）
dataset = dataset.map(tokenize_dataset, batched=True)

# 用来从数据集中创建批次的 [DataCollatorWithPadding]
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
    data_collator=data_collator,
)  # doctest: +SKIP

# 对于像翻译或摘要这些使用序列到序列模型的任务,用 `Seq2SeqTrainer` 和 `Seq2SeqTrainingArguments` 来替代
trainer.train()
```