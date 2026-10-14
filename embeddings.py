
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read();

#print("Total characters: ", len(raw_text), "\n", raw_text[:99])

import re

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text);
preprocessed = [ item.strip() for item in preprocessed if item.strip()]
#print(len(preprocessed), "\n", preprocessed[:30])


# paragraphs -> vocab
all_words = sorted(set(preprocessed))
#print(len(all_words))

vocab = {token:integer for token, integer in enumerate(all_words)}

for i, item in enumerate(vocab.items()):
#   print(item)
    if i > 50:
        break;

from importlib.metadata import version
import tiktoken
print("tiktoken version:", version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, Ashutosh")
integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(integers)

