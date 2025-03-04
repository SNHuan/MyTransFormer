# MyTransformer

一个基于 Transformer 架构的中文语言模型实现。

## 项目结构

- `modeling/`: 模型核心实现
  - `modeling.py`: 包含 Transformer 模型的核心实现，包括自注意力机制、MLP、解码器层等
  - `config.py`: 模型配置类，定义了模型的超参数

- `custom_train.py`: 模型训练脚本，支持从头开始训练
- `continue_train.py`: 从检查点继续训练的脚本
- `better_generate.py`: 文本生成脚本，用于使用训练好的模型生成文本

## 功能特点

- 实现了完整的 Transformer 解码器架构
- 支持 RMSNorm 归一化
- 实现了 SwiGLU 激活函数
- 支持 KV 缓存以加速推理
- 支持混合精度训练
- 与 🤗 Transformers 库完全兼容

## 使用方法

### 训练模型

```bash
python custom_train.py \
    --model_size nano \
    --max_steps 100 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 32 \
    --fp16 \
    --tokenizer_path ./models/new_tokenizer \
    --dataset_path data/chinese_deepseek_r1
```

### 继续训练

```bash
python continue_train.py \
    --checkpoint_path ./output/checkpoint-1000 \
    --tokenizer_path ./models/new_tokenizer \
    --max_steps 1000 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --max_length 32 \
    --fp16
```

### 生成文本

```bash
python better_generate.py \
    --model_path ./output/checkpoint-10000 \
    --tokenizer_path models/new_tokenizer \
    --prompt "你好" \
    --max_length 50 \
    --temperature 0.7 \
    --top_p 0.9 \
    --top_k 100
```

## 模型配置

支持两种预设配置：

- nano:
  - hidden_size: 256
  - num_hidden_layers: 4
  - num_attention_heads: 4
  - intermediate_size: 1024

- medium:
  - hidden_size: 768
  - num_hidden_layers: 12
  - num_attention_heads: 12
  - intermediate_size: 3072

## 依赖

- PyTorch
- Transformers
- Datasets
- NumPy
