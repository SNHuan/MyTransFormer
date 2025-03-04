import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm
from modeling.config import TestConfig
from typing import Optional, Tuple
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.generation import GenerationMixin
from torch.nn.functional import scaled_dot_product_attention

# 定义RMSNorm层，用于归一化
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化权重参数

    def forward(self, hidden_states):
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)  # 计算方差
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)  # 归一化
        
        return self.weight * hidden_states  # 返回归一化后的结果乘以权重


# 定义残差连接层
class Residual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # 初始化权重参数
        
    def forward(self, residual, hidden_states):
        return self.weight * residual + hidden_states  # 返回残差连接的结果


# 定义自注意力层
class SelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.hidden_dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.k_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.v_proj = nn.Linear(self.hidden_dim, self.num_heads * self.head_dim)
        self.o_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        
    def forward(self, hidden_states, attention_mask, past_key_values=None):
        batch_size, seq_length = hidden_states.shape[:2]
        
        # Project and reshape
        q = self.q_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_length, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn_output = scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attention_mask,
            dropout_p=0.1 if self.training else 0
        )
        
        # Reshape and project back
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_dim)
        return self.o_proj(attn_output)
        
    
# 定义多层感知器（MLP）
class MLP(nn.Module):
    def __init__(self, dim, ffn_dim):
        super().__init__()
        self.hidden_dim = dim
        self.ffn_dim = ffn_dim
        self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim)
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states):
        gate_output = self.gate_proj(hidden_states)
        up_output = self.up_proj(hidden_states)
        gate_up = self.act_fn(gate_output) * up_output
        output = self.down_proj(gate_up)
        return output


# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self, dim, num_heads, ffn_dim):
        super().__init__()
        self.pre_norm = LlamaRMSNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads)
        self.post_norm = LlamaRMSNorm(dim)
        self.feed_forward = MLP(dim, ffn_dim)

    def forward(self, hidden_states, attention_mask, past_key_values=None):
        # Self Attention
        residual = hidden_states
        hidden_states = self.pre_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, past_key_values)
        hidden_states = residual + hidden_states

        # Feed Forward
        residual = hidden_states
        hidden_states = self.post_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
    
class TestPreTrainedModel(PreTrainedModel):
    config_class = TestConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DecoderLayer"]
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

class TestModel(TestPreTrainedModel):
    def __init__(self, config: TestConfig):
        """
        初始化TestModel模型
        Args:
            config: 模型配置对象
        """
        super().__init__(config)
        self.config = config
        
        # 词嵌入层，将词表示转换为向量
        self.word_embed = nn.Embedding(config.vocab_size, config.hidden_size)
        # 位置嵌入层，为每个位置添加位置信息
        self.position_embed = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建多层解码器
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(config.hidden_size, config.num_attention_heads, config.intermediate_size) 
            for _ in range(config.num_hidden_layers)
        ])
        # 最终的归一化层
        self.norm = LlamaRMSNorm(config.hidden_size)
        # 梯度检查点标志
        self.gradient_checkpointing = False
        self.post_init()
        
    def get_input_embeddings(self):
        """获取输入嵌入层"""
        return self.word_embed

    def set_input_embeddings(self, value):
        """设置输入嵌入层"""
        self.word_embed = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
    ) -> BaseModelOutputWithPast:
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        hidden_states = self.word_embed(input_ids)
        position_ids = torch.arange(0, input_ids.shape[1], device=input_ids.device)
        hidden_states = hidden_states + self.position_embed(position_ids)
        
        batch_size, seq_length = input_ids.shape
        causal_mask = torch.triu(
            torch.full((seq_length, seq_length), float("-inf"), device=input_ids.device),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # [1, 1, seq_length, seq_length]
        
        if attention_mask is not None:
            attention_mask = attention_mask.view(batch_size, 1, 1, seq_length)
            attention_mask = attention_mask.expand(-1, 1, seq_length, -1)
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask + causal_mask
        else:
            attention_mask = causal_mask.expand(batch_size, -1, -1, -1)
            
        current_past_key_values = []
        
        for idx, decoder_layer in enumerate(self.decoder_layers):
            layer_past = None
            if past_key_values is not None:
                layer_past = past_key_values[idx]
                
            hidden_states = decoder_layer(hidden_states, attention_mask, layer_past)
            
            if hasattr(decoder_layer.self_attn, 'kv_cache'):
                current_past_key_values.append(decoder_layer.self_attn.kv_cache)
            
        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=tuple(current_past_key_values) if current_past_key_values else None
        )

class TestForCausalLM(TestPreTrainedModel, GenerationMixin):
    """
    用于因果语言建模的模型类
    继承自PreTrainedModel和GenerationMixin
    """
    def __init__(self, config: TestConfig):
        super().__init__(config)
        self.vocab_size = config.vocab_size
        self.model = TestModel(config)
        # 使用词嵌入的转置作为语言模型头部
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 设置权重共享
        self.lm_head.weight = self.model.word_embed.weight
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        """
        前向传播函数
        Args:
            input_ids: 输入的token ID序列
            attention_mask: 注意力掩码
            labels: 目标标签
            past_key_values: 过去的键值对，用于加速生成
            return_dict: 是否返回字典格式的输出
        Returns:
            包含logits和损失的输出对象
        """
        # 如果有过去的键值对，只处理最后一个token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        # 获取基础模型的输出，传递past_key_values
        outputs = self.model(input_ids, attention_mask, past_key_values)
        logits = self.lm_head(outputs.last_hidden_state)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
            
        if return_dict is None:
            return_dict = self.config.use_return_dict
            
        if not return_dict:
            return (logits,) + (loss,) if loss is not None else logits
            
        return CausalLMOutputWithPast(
            logits=logits, 
            past_key_values=outputs.past_key_values,
            loss=loss
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        """
        为生成准备输入
        Args:
            input_ids: 输入的token ID序列
            past_key_values: 过去的键值对，用于加速生成
            kwargs: 额外的参数
        Returns:
            准备好的输入字典
        """
        attention_mask = kwargs.get("attention_mask", None)
        
        # 如果有过去的键值对，调整输入
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
        }

    def get_output_embeddings(self):
        """获取输出嵌入层"""
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        """设置输出嵌入层"""
        self.lm_head = new_embeddings
