import os
import logging
import torch
from datasets import load_dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)
from modeling.config import TestConfig
from modeling.modeling import TestForCausalLM

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def load_or_download_dataset(dataset_name, dataset_path, local_files_only=False):
    """Load dataset from local path or download from Hugging Face"""
    if os.path.exists(dataset_path):
        logger.info(f"Loading dataset from {dataset_path}")
        return load_from_disk(dataset_path)
    
    if local_files_only:
        raise ValueError(f"Dataset path {dataset_path} does not exist and local_files_only is set")
    
    logger.info(f"Downloading dataset {dataset_name}")
    os.makedirs(dataset_path, exist_ok=True)
    ds = load_dataset(dataset_name)
    ds.save_to_disk(dataset_path)
    return ds

def get_model_config(size, vocab_size):
    """Get model configuration based on size"""
    configs = {
        "nano": {
            "hidden_size": 256,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "intermediate_size": 1024,
        },
        "medium": {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
        }
    }
    
    if size not in configs:
        logger.warning(f"Unknown model size: {size}, using nano config")
        size = "nano"
        
    config = configs[size]
    return TestConfig(
        vocab_size=vocab_size,
        max_position_embeddings=2048,
        **config
    )

def get_text_column(dataset):
    """Determine which column contains the text data"""
    if not dataset:
        raise ValueError("Dataset is empty")
        
    sample = dataset[0]
    required_columns = ["input", "reasoning_content", "content"]
    
    # 检查必需的列是否都存在
    for col in required_columns:
        if col not in sample:
            raise ValueError(f"Required column '{col}' not found in dataset")
            
    return required_columns

def main(args):
    set_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=args.local_files_only,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")
    
    # Load dataset
    dataset = load_or_download_dataset(args.dataset_name, args.dataset_path, args.local_files_only)
    text_columns = get_text_column(dataset["train"])
    logger.info(f"Using columns {text_columns} for training")
    
    # 显示数据集信息
    logger.info(f"Dataset size: {len(dataset['train'])} samples")
    logger.info(f"Dataset columns: {dataset['train'].column_names}")
    
    # 显示样本示例
    sample = dataset["train"][0]
    logger.info("Sample data:")
    for key in text_columns:
        logger.info(f"{key}: {sample[key][:100]}...")
    
    def preprocess_function(examples):
        texts = []
        
        # 处理每个样本
        for i in range(len(examples["input"])):
            text_parts = []
            
            # 按顺序添加各个部分
            if examples["input"][i]:
                text_parts.append(examples["input"][i])
            if examples["reasoning_content"][i]:
                text_parts.append(examples["reasoning_content"][i])
            if examples["content"][i]:
                text_parts.append(examples["content"][i])
            
            # 使用换行符连接
            full_text = "\n".join(text_parts)
            texts.append(full_text)
        
        # 记录处理示例
        if len(texts) > 0 and args.debug:
            logger.info("Preprocessed sample:")
            logger.info(f"{texts[0][:500]}...")
        
        # 分词处理
        tokenized = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        
        # 设置标签
        tokenized["labels"] = tokenized["input_ids"].clone()
        
        return tokenized
    
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names,
    )
    
    # Initialize model
    config = get_model_config(args.model_size, len(tokenizer))
    model = TestForCausalLM(config)
    model.to(device)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=100,
        save_steps=1000,
        fp16=args.fp16 and device == "cuda",
        seed=args.seed,
        save_safetensors=False,  # 使用更安全的保存格式
        save_strategy="steps",
        save_total_limit=4,
        no_cuda=args.use_cpu,
        report_to="none",  # 不报告到任何平台
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
    )
    
    # Train and save
    logger.info("Starting training")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Model and tokenizer saved to {args.output_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train a custom Transformer model")
    parser.add_argument("--dataset_name", type=str, help="Dataset name on HuggingFace")
    parser.add_argument("--dataset_path", type=str, default="data/dataset", help="Local dataset path")
    parser.add_argument("--model_size", type=str, default="nano", choices=["nano", "medium"])
    parser.add_argument("--max_length", type=int, default=32, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=100, help="Maximum training steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--fp16", action="store_true", help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
    parser.add_argument("--tokenizer_path", type=str, default="models/tokenizer")
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    main(args) 