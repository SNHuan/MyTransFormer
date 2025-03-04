import os
import sys
import argparse
import logging
import torch
import numpy as np
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    get_scheduler,
)

# 导入自定义模型和配置
from modeling.modeling import TestForCausalLM
from modeling.config import TestConfig

#python continue_train.py --checkpoint_path ./output/checkpoint-1000 --tokenizer_path ./models/new_tokenizer --max_steps 1000 --batch_size 1 --gradient_accumulation_steps 16 --max_length 32 --fp16

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_or_download_dataset(dataset_name, dataset_path):
    """从本地加载或从Hugging Face下载数据集"""
    if os.path.exists(dataset_path):
        logger.info("从本地加载数据集...")
        ds = load_from_disk(dataset_path)
    else:
        logger.info("从Hugging Face下载数据集...")
        os.makedirs(dataset_path, exist_ok=True)
        ds = load_dataset(dataset_name)
        ds.save_to_disk(dataset_path)
        logger.info("数据集已保存到本地")
    
    return ds

def parse_args():
    parser = argparse.ArgumentParser(description="从检查点继续训练自定义Transformer模型")
    
    # 数据集参数
    parser.add_argument("--dataset_name", type=str, default=None, help="Hugging Face上的数据集名称")
    parser.add_argument("--dataset_path", type=str, default="data/chinese_deepseek_r1", help="数据集路径")
    
    # 模型参数
    parser.add_argument("--checkpoint_path", type=str, required=True, help="检查点路径")
    parser.add_argument("--max_length", type=int, default=32, help="最大序列长度")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=1, help="训练批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max_steps", type=int, default=5000, help="最大训练步数")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="梯度累积步数")
    parser.add_argument("--fp16", action="store_true", help="是否使用混合精度训练")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output_dir", type=str, default="./output", help="输出目录")
    parser.add_argument("--tokenizer_path", type=str, default="models/new_tokenizer", help="分词器路径")
    parser.add_argument("--local_files_only", action="store_true", default=True, help="仅使用本地文件")
    parser.add_argument("--use_cpu", action="store_true", help="强制使用CPU")
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    logger.info(f"使用设备: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"可用GPU数量: {torch.cuda.device_count()}")
    
    # 检查路径是否存在
    tokenizer_path = os.path.abspath(args.tokenizer_path)
    checkpoint_path = os.path.abspath(args.checkpoint_path)
    
    logger.info(f"检查分词器路径: {tokenizer_path}")
    if not os.path.exists(tokenizer_path):
        logger.warning(f"警告: 分词器路径 {tokenizer_path} 不存在")
        # 尝试在当前目录下查找
        if os.path.exists(os.path.join(os.getcwd(), args.tokenizer_path)):
            tokenizer_path = os.path.join(os.getcwd(), args.tokenizer_path)
            logger.info(f"找到分词器路径: {tokenizer_path}")
    
    logger.info(f"检查检查点路径: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.warning(f"警告: 检查点路径 {checkpoint_path} 不存在")
        # 尝试在当前目录下查找
        if os.path.exists(os.path.join(os.getcwd(), args.checkpoint_path)):
            checkpoint_path = os.path.join(os.getcwd(), args.checkpoint_path)
            logger.info(f"找到检查点路径: {checkpoint_path}")
    
    # 加载分词器
    logger.info(f"加载分词器: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=args.local_files_only
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("设置 pad_token 为 eos_token")
    except Exception as e:
        logger.error(f"加载分词器失败: {e}")
        raise
    
    logger.info(f"分词器词表大小: {len(tokenizer)}")
    
    # 加载数据集
    logger.info(f"加载数据集: {args.dataset_path}")
    dataset = load_or_download_dataset(args.dataset_name, args.dataset_path)
    logger.info(f"数据集大小: {len(dataset['train'])} 样本")
    
    # 检查数据集结构
    logger.info(f"数据集列名: {dataset['train'].column_names}")
    
    # 获取一个样本进行检查
    sample = dataset['train'][0]
    logger.info("数据集样本示例:")
    for key, value in sample.items():
        if isinstance(value, str):
            logger.info(f"  {key}: {value[:100]}...")  # 只显示前100个字符
        else:
            logger.info(f"  {key}: {value}")
    
    # 从检查点加载模型
    logger.info(f"从检查点加载模型: {checkpoint_path}")
    try:
        model = TestForCausalLM.from_pretrained(
            checkpoint_path,
            local_files_only=args.local_files_only,
            trust_remote_code=True
        )
        model.resize_token_embeddings(len(tokenizer))
        logger.info("调整模型词表大小以匹配分词器")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise
    
    # 将模型移动到设备
    model.to(device)
    
    logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    
    # 数据预处理
    def tokenize_function(examples):
        texts = []
        
        # 处理Chinese-DeepSeek-R1数据集的特定结构
        for i in range(len(examples["input"])):
            # 按照input -> reasoning_content -> content的顺序拼接
            text_parts = []
            
            # 添加输入部分
            if examples["input"][i]:
                text_parts.append(examples["input"][i])
            
            # 添加推理过程
            if examples["reasoning_content"][i]:
                text_parts.append(examples["reasoning_content"][i])
            
            # 添加输出内容
            if examples["content"][i]:
                text_parts.append(examples["content"][i])
            
            # 使用换行符连接各部分
            full_text = "\n".join(text_parts)
            texts.append(full_text)
        
        if not texts:
            raise ValueError("数据集处理失败：无法找到有效的文本")
        
        # 记录一些样本示例
        if len(texts) > 0:
            logger.info("数据处理示例:")
            logger.info(f"处理后的第一个样本:\n{texts[0][:500]}...")
            
        # 分词
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        
        # 准备语言模型的标签
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    # 对数据集进行分词
    logger.info("对数据集进行分词")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    
    # 数据收集器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因为我们是做因果语言模型训练
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=10,
        save_steps=1000,
        fp16=args.fp16 and device == "cuda",  # 只有在使用GPU时才启用fp16
        seed=args.seed,
        save_safetensors=False,  # 使用更安全的保存格式
        save_strategy="steps",
        save_total_limit=10,
        no_cuda=args.use_cpu,
        eval_strategy="no",  # 移除评估，因为我们没有验证集
        report_to="none"
    )
    
    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始继续训练")
    trainer.train(resume_from_checkpoint=checkpoint_path)
    
    # 保存模型和分词器
    logger.info(f"保存模型和分词器到 {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    logger.info("训练完成")

if __name__ == "__main__":
    main() 