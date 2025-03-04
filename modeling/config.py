from transformers import PretrainedConfig

class TestConfig(PretrainedConfig):
    model_type = "test"
    
    def __init__(
        self,
        vocab_size=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        max_position_embeddings=2048,
        hidden_dropout_prob=0.1,
        attention_dropout_prob=0.1,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        tie_word_embeddings=True,
        **kwargs
    ):
        """
        Args:
            vocab_size (int): 词表大小
            hidden_size (int): 隐藏层维度
            num_hidden_layers (int): 解码器层数量
            num_attention_heads (int): 注意力头数量
            intermediate_size (int): 前馈网络中间层维度
            max_position_embeddings (int): 最大位置编码长度
            hidden_dropout_prob (float): 隐藏层dropout概率
            attention_dropout_prob (float): 注意力层dropout概率
            initializer_range (float): 权重初始化范围
            rms_norm_eps (float): RMSNorm的epsilon值
            use_cache (bool): 是否使用KV缓存
            pad_token_id (int): padding token的ID
            bos_token_id (int): 序列开始token的ID
            eos_token_id (int): 序列结束token的ID
            tie_word_embeddings (bool): 是否共享输入输出词嵌入
        """
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
