import os

def get_model_path():
    return os.getenv("MODEL_PATH", "default-model-path")

def get_tokenizer_path():
    return os.getenv("TOKENIZER_PATH", "default-tokenizer-path")
