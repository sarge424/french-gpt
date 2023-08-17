import torch

from transformer import Transformer
from tokenizer import BPEncoder

sentences = [
    "hi!"
]

device= 'cuda'

def split(line):
    words = []
    curr = ''
    for ch in line.lower():
        if ch == ' ':
            words.append(curr+'_')
            curr = ''
        elif ch.isalpha() or ch == '\'':
            curr += ch
        else:
            words += [curr+'_', ch]
            curr = ''
    return words

sentences = [split(sen) for sen in sentences]

bpe = BPEncoder()
bpe.load_tokens('langgpt/data/tokens.json')

tokensen = []
for line in sentences:
    tokens = ['^']
    for word in line:
        tokens += bpe.tokenize(word)
    tokens += ['/']
    tokensen.append(tokens)

x = bpe.encode(tokensen)
print('encoded input:', x)

model = torch.load('langgpt/models/model.pt')
model = model.to(device)
model.eval()

inp = torch.tensor(x[0], dtype=torch.long).to(device)
y = model.generate(inp.view(1, -1), 0)

print(' '.join(bpe.decode([y[0].tolist()])[0]))#.replace('_', ' '))