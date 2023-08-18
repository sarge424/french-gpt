import torch

from transformer import Transformer
from tokenizer import BPEncoder

sentences = [
    "I need to buy some groceries.",
    "He is studying hard for his exams.",
    "Can you please pass me the salt?",
    "She is learning to play the piano.",
    "We enjoyed a delicious dinner at the restaurant."
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
bpe.load_tokens('langgpt/data/tokens750.json')

tokensen = []
for line in sentences:
    tokens = ['^']
    for word in line:
        tokens += bpe.tokenize(word)
    tokens += ['/']
    tokensen.append(tokens)

x = bpe.encode(tokensen)
#print('encoded input:', x)

model = torch.load('langgpt/models/model.pt')
model = model.to(device)
model.eval()

for row in x:
    inp = torch.tensor(row, dtype=torch.long).to(device)
    y = model.generate(inp.view(1, -1), 0)

    #print(' '.join(bpe.decode([y[0].tolist()])[0]))#.replace('_', ' '))
    output = ''.join(bpe.decode([y[0].tolist()])[0]).replace('_', ' ')
    output = '\n'.join(output[1:-1].split('/'))
    print(output, '\n')