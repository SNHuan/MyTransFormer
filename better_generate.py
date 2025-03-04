import os
import torch
import argparse
import logging
from transformers import AutoTokenizer
from modeling.modeling import TestForCausalLM
#python better_generate.py --model_path ./output/checkpoint-10000 --tokenizer_path models/new_tokenizer --prompt "你好" --max_length 50 --temperature 0.7 --top_p 0.9 --top_k 100 --num_return_sequences 1 --debug

# 设置日志
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的模型生成文本")
    parser.add_argument("--model_path", type=str, default="./output", help="模型路径")
    parser.add_argument("--tokenizer_path", type=str, default="models/new_tokenizer", help="分词器路径")
    parser.add_argument("--prompt", type=str, required=True, help="输入提示")
    parser.add_argument("--max_length", type=int, default=50, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.7, help="温度参数")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p采样参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k采样参数")
    parser.add_argument("--num_return_sequences", type=int, default=1, help="返回序列数量")
    parser.add_argument("--debug", action="store_true", help="是否显示调试信息")
    parser.add_argument("--use_cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--local_files_only", action="store_true", default=True, help="仅使用本地文件")
    parser.add_argument("--repetition_penalty", type=float, default=1.1, help="重复惩罚参数")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="禁止重复的N元组大小")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置设备
    device = "cuda" if torch.cuda.is_available() and not args.use_cpu else "cpu"
    logger.info(f"使用设备: {device}")
    if device == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # 检查路径是否存在
    tokenizer_path = os.path.abspath(args.tokenizer_path)
    model_path = os.path.abspath(args.model_path)
    
    logger.info(f"检查分词器路径: {tokenizer_path}")
    if not os.path.exists(tokenizer_path):
        logger.warning(f"警告: 分词器路径 {tokenizer_path} 不存在")
        if os.path.exists(os.path.join(os.getcwd(), args.tokenizer_path)):
            tokenizer_path = os.path.join(os.getcwd(), args.tokenizer_path)
            logger.info(f"找到分词器路径: {tokenizer_path}")
    
    logger.info(f"检查模型路径: {model_path}")
    if not os.path.exists(model_path):
        logger.warning(f"警告: 模型路径 {model_path} 不存在")
        if os.path.exists(os.path.join(os.getcwd(), args.model_path)):
            model_path = os.path.join(os.getcwd(), args.model_path)
            logger.info(f"找到模型路径: {model_path}")
    
    # 加载分词器
    logger.info(f"加载分词器: {tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            local_files_only=args.local_files_only,
            trust_remote_code=True
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("设置 pad_token 为 eos_token")
    except Exception as e:
        logger.error(f"加载分词器失败: {e}")
        logger.info("尝试从模型路径加载分词器...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                local_files_only=args.local_files_only,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("设置 pad_token 为 eos_token")
        except Exception as e:
            logger.error(f"从模型路径加载分词器失败: {e}")
            raise
    
    # 显示分词器信息
    if args.debug:
        logger.info(f"分词器词表大小: {len(tokenizer)}")
        logger.info(f"特殊标记: {tokenizer.all_special_tokens}")
        logger.info(f"特殊标记ID: {tokenizer.all_special_ids}")
    
    # 加载模型
    logger.info(f"加载模型: {model_path}")
    try:
        model = TestForCausalLM.from_pretrained(
            model_path,
            local_files_only=args.local_files_only,
            trust_remote_code=True
        )
        model.resize_token_embeddings(len(tokenizer))
        logger.info("调整模型词表大小以匹配分词器")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        raise
    
    model.to(device)
    model.eval()
    
    # 编码输入
    logger.info(f"\n输入提示: {args.prompt}")
    input_ids = tokenizer.encode(args.prompt, return_tensors="pt").to(device)
    
    if args.debug:
        logger.info(f"输入ID: {input_ids.tolist()}")
        logger.info(f"输入标记: {[tokenizer.decode([id]) for id in input_ids[0].tolist()]}")
    
    # 设置生成参数
    gen_kwargs = {
        "max_length": args.max_length,
        "num_return_sequences": args.num_return_sequences,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "do_sample": True,
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": args.repetition_penalty,
        "no_repeat_ngram_size": args.no_repeat_ngram_size,
        "early_stopping": True,
    }
    
    # 生成文本
    logger.info("\n开始生成...")
    try:
        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                **gen_kwargs
            )
        
        # 解码并显示结果
        for i, output in enumerate(outputs):
            generated_text = tokenizer.decode(output, skip_special_tokens=True)
            logger.info(f"\n生成结果 {i+1}:")
            logger.info(generated_text)
            
            if args.debug:
                logger.info(f"\n生成的ID序列:")
                logger.info(output.tolist())
                logger.info(f"\n生成的标记序列:")
                for id in output.tolist():
                    token = tokenizer.decode([id])
                    logger.info(f"ID: {id}, 标记: '{token}'")
    
    except Exception as e:
        logger.error(f"生成过程中出错: {e}")
        raise

if __name__ == "__main__":
    main() 