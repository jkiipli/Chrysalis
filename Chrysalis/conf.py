
GPT_CONFIG_V11 = {
    "version": 11,
    "lang":"_l4",
    "tokenizer": "bpe",
    "logits": True,
    "vocab_size": 1000,    # Vocabulary size
    "context_length": 60,  # Shortened context length (orig: 1024)
    "emb_dim": 200,        # Embedding dimension
    "n_heads": 10,          # Number of attention heads
    "n_layers": 5,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}
GPT_CONFIG_V12 = {
    "version": 12,
    "lang":"_l4",
    "tokenizer": "bpe",
    "logits": True,
    "vocab_size": 1000,    # Vocabulary size
    "context_length": 60,  # Shortened context length (orig: 1024)
    "emb_dim": 200,        # Embedding dimension
    "n_heads": 10,          # Number of attention heads
    "n_layers": 3,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}
GPT_CONFIG_V13 = {
    "version": 13,
    "lang":"_l4",
    "tokenizer": "bpe",
    "logits": True,
    "vocab_size": 1000,    # Vocabulary size
    "context_length": 60,  # Context length
    "emb_dim": 100,        # Embedding dimension
    "n_heads": 10,          # Number of attention heads
    "n_layers": 3,         # Number of layers
    "drop_rate": 0.,       # Dropout rate
    "qkv_bias": False       # Query-key-value bias
}

confs = [GPT_CONFIG_V11,GPT_CONFIG_V12,GPT_CONFIG_V13]

def get_conf(filename):
    for c in confs:
        if "_v"+str(c["version"]) in filename: return c
    return None