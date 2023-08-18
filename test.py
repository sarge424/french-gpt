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


bpe = BPEncoder()
bpe.load_tokens('langgpt/data/tokens750.json')

model = torch.load('langgpt/models/model.pt')
model = model.to(device)
model.eval()


while True:
    #get sentence
    sentence = input('ENG: ')
    
    if sentence == 'exit':
        break
    sentences = [split(sentence)]

    #tokenize
    tokensen = []
    for line in sentences:
        tokens = ['^']
        for word in line:
            tokens += bpe.tokenize(word)
        tokens += ['/']
        tokensen.append(tokens)

    #encode
    x = bpe.encode(tokensen)
    
    #generate
    inp = torch.tensor(x[0], dtype=torch.long).to(device)
    y = model.generate(inp.view(1, -1), 0)

    #print result
    output = ''.join(bpe.decode([y[0].tolist()])[0]).replace('_', ' ')
    output = output[1:-1].split('/')[1]
    print('FRE:', output, '\n')