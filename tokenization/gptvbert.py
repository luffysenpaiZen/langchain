from transformers import AutoTokenizer

tokenizer_gpt=AutoTokenizer.from_pretrained('gpt2')
tokenizer_bert=AutoTokenizer.from_pretrained('bert-base-uncased')

tokens_gpt=tokenizer_gpt.tokenize("Hi my name is issac")


tokens_bert=tokenizer_bert.tokenize("Hi my name is issac")

