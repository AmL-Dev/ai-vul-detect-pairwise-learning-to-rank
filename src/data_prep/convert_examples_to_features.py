import json
from transformers import PreTrainedTokenizer

def convert_examples_to_features(js: json, tokenizer: PreTrainedTokenizer):
    code = js['func']
    source_tokens = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    return source_tokens