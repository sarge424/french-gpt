import torch
import random
from datetime import datetime

from tokenizer import BPEncoder
from transformer import Transformer

#HYPERPARAMETERS:---------------------------------------------
device = 'cuda'
n_heads = 4
n_channels = 32
n_layers = 2
block_size = 200
batch_size = 4
dropout = 0.0



#Read and encode the data
print('reading data...')
data = open('langgpt/data/data_e750.txt', 'r', encoding='utf-8')
data = [['^'] + line.split() + ['*'] for line in data.read().splitlines()[:101]]

print('encoding...')
bpe = BPEncoder()
bpe.load_tokens('langgpt/data/tokens750.json')
data = bpe.encode(data)


#HYPERPARAMETERS:---------------------------------------------
mlen = max(len(line) for line in data)
print('max len is', mlen)
vsize = len(bpe.tokens)
max_iters = 5000
print('max iters is', max_iters)
learning_rate = 1e-2

val_loss_iters = 100
training_batches = 500
val_size = 20
test_size = 1

#Split the dataset into examples of block_size
print('creating dataset...', end='')
x, y = [],[]
for line in data:
    assert 2 in line, 'no / in ' + ' '.join(str(i) for i in line)
    line = ([0] * max(mlen, block_size)) + line
    sp = line.index(2) + 1
    st = sp-block_size
    for i in range(st, len(line)-block_size):
        en = i + block_size
        x.append(line[i:en])
        y.append(line[i+1:en+1])
      
x = torch.tensor(x, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

indices = [i for i in range(x.shape[0])]
random.shuffle(indices)
itr = torch.tensor(indices[:-(val_size + test_size)], dtype=torch.long)
ival = torch.tensor(indices[-val_size:-test_size], dtype=torch.long)
itest = torch.tensor(indices[-test_size:], dtype=torch.long)

xtr, ytr = x[itr], y[itr]
xval, yval = x[ival], y[ival]

print(f'train: {len(itr)} val: {len(ival)} test: {len(itest)}')


#Helper Functions
def get_batch(split):
    xs, ys = (xtr, ytr) if split == 'train' else (xval, yval)
    idx = torch.randint(0, xs.shape[0]-1, (batch_size, ))
    xb, yb = xs[idx], ys[idx]
    
    crop = 0
    while crop < xb.shape[1]:
        if torch.all(xb[:, crop] == 0) and torch.all(yb[:, crop] == 0):
            crop += 1
        else:
            return xb[:, crop:].to(device), yb[:, crop:].to(device)
    
    raise Exception('all 0 xb...' + str(xb))

@torch.no_grad()
def val_loss():
    losses = []
    for _ in range(val_loss_iters):
        xb, yb = get_batch('val')
        _, loss = model(xb, yb)
        losses.append(loss.item())
    return sum(losses) / len(losses)


#Create the model
print('creating model...', end='')
model = Transformer(
    heads=n_heads,
    channels=n_channels,
    growth=4,
    dropout=dropout,
    depth=n_layers,
    vsize=vsize,
    mlen=block_size,
    device=device
).to(device)
print('created (', model.num_params(), ' params).', sep='')
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


#Train the model
model.train()
losses = []
for i in range(max_iters):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if i % training_batches == 0 or i == max_iters-1:
        time = datetime.now().strftime('%H:%M:%S')
        print(time ,i, sum(losses)/len(losses), val_loss())
        losses = []


#Save the model
print('saving model...')
torch.save(model, 'langgpt/models/model.pt')
