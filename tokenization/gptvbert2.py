
from transformers import GPT2Tokenizer, BertTokenizer
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
import matplotlib.pyplot as plt
from collections import Counter

sentence = "I can't believe it's already 2025!"



gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# GPT2
print(len(sentence))
gpt2_tokens = gpt2_tokenizer.tokenize (sentence)
gpt2_ids = gpt2_tokenizer.encode(sentence)
gpt2_decoded = gpt2_tokenizer.decode(gpt2_ids)
print(f"GPT2 Tokens:{len(gpt2_tokens)}", gpt2_tokens)
print("GPT2 Token IDs:", gpt2_ids)
print("GPT2 Decoded:", gpt2_decoded)
# BERT
bert_tokens = bert_tokenizer.tokenize (sentence)
bert_ids = bert_tokenizer.encode(sentence)
bert_decoded = bert_tokenizer.decode(bert_ids)
print(f"\nBERT Tokens:{len(bert_tokens)}", bert_tokens)
print("BERT Token IDs:", bert_ids)
print("BERT Decoded:", bert_decoded)

sentences = [
"Hello!",
"The quick brown fox jumps over the lazy dog.",
"Machine learning models require lots of data.",
"Antidisestablishmentarianism is a long word.",
"Supercalifragilisticexpialidocious is even longer!"]
print(f"{'Sentence':50} {'Chars':>5} {'Tokens':>6}")
print("-"*65)
for s in sentences:
    num_chars = len(s)
    num_tokens = len(gpt2_tokenizer.encode(s))
    print(f"{s:50} {num_chars:5} {num_tokens:6}")
    

# Create toy corpus
corpus = [
    "the quick brown fox jumped over the lazy dog",
    "never jump over the lazy dog quickly",
    "the fox is quick and the dog is lazy"
]
# Save corpus to file
with open("toy_corpus.txt", "w") as f:
    for line in corpus:
        f.write(line + "\n")
# Initialize tokenizer
custom_tokenizer = Tokenizer(models.BPE())
custom_tokenizer1=Whitespace()
trainer1=BpeTrainer()
custom_tokenizer.train(["toy_corpus.txt"], trainer1)
output = custom_tokenizer.encode("the quickfox is lazy")
print("Custom Tokens:", output.tokens)
print(([len(set(corpus[i])) for i in range(3)]))


# Pretend the ID list has an OOV
token_ids = [464, 389, 3456, 9999999]
#GPT2 vocab size
vocab_size = gpt2_tokenizer.vocab_size
safe_token_ids = [tid if tid < vocab_size else gpt2_tokenizer.unk_token_id for tid in token_ids]
print("Original IDs: ", token_ids)
print("Safe IDs: ", safe_token_ids)
print("Decoded:", gpt2_tokenizer.decode(safe_token_ids))



text = """"
In computer science, tokenization is the process of breaking text up into words, phrases, symbols, or other meaningful elements called tokens.
The goal of tokenization is to explore the meaning and structure of the text."""
tokens = gpt2_tokenizer.tokenize(text)
print(len(tokens))
freqs = Counter(tokens)
print(len(freqs))
print("Most common tokens:", freqs.most_common(10))
# Plot
sum1=0
for key,val in freqs.most_common(10):
    sum1+=val
print(sum1)
most_common = freqs.most_common(10)
labels, counts = zip(*most_common)
plt.bar(labels, counts)
plt.xticks(rotation=45)
plt.title("Top 10 Tokens")
plt.show()


from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt-40") # or any model
text = "your very long text here..." # 20,000 tokens
tokens = tokenizer.encode(text)
print(f"Total tokens: {len(tokens)}")
max_context = 16384
if len(tokens) > max_context:
    tokens = tokens[:max_context]
    truncated_text = tokenizer.decode(tokens)
    print(f"Truncated to {len(tokens)} tokens.")