
from transformers import AutoTokenizer

model_name = "BAAI/bge-large-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Len tokenizer: {len(tokenizer)}")

if len(tokenizer) > tokenizer.vocab_size:
    print("WARNING: Tokenizer has added tokens!")
    print(f"Added tokens: {len(tokenizer) - tokenizer.vocab_size}")
