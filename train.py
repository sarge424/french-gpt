import torch
from datetime import datetime

from tokenizer import BPEncoder
from transformer import Transformer

#HYPERPARAMETERS:---------------------------------------------
device = 'cuda'
n_heads = 8
n_channels = 512
n_layers = 8
block_size = 200
batch_size = 32
dropout = 0.1



#Read and encode the data
print('reading data...')
data = open('langgpt/data/data_e750.txt', 'r', encoding='utf-8')
data = [['^'] + line.split() + ['*'] for line in data.read().splitlines()]

print('encoding...')
bpe = BPEncoder()
bpe.load_tokens('langgpt/data/tokens750.json')
data = bpe.encode(data)


#HYPERPARAMETERS:---------------------------------------------
mlen = max(len(line) for line in data)
print('max len is', mlen)
vsize = len(bpe.tokens)
max_iters = 50000
print('max iters is', max_iters)
learning_rate = 3e-4

val_loss_iters = 50
training_batches = 1000
val_size = 3000
test_size = 3000

#Split the dataset into examples of block_size
print('creating datasets...')

train_data = data[:-(val_size + test_size)]
val_data = data[-(val_size + test_size):-test_size]
test_data = data[-test_size:]

#Helper Functions
def get_batch(split):
    d = train_data if split == 'train' else val_data
    idx = torch.randint(0, len(d)-1, (batch_size, ))
    
    xs = [d[i][0:block_size] for i in idx]
    ys = [d[i][1:block_size+1] for i in idx]
    
    xs = [line + ([0] * (block_size - len(line))) for line in xs]
    ys = [line + ([0] * (block_size - len(line))) for line in ys]
    
    xb = torch.tensor(xs, dtype=torch.long).to(device)
    yb = torch.tensor(ys, dtype=torch.long).to(device)

    return xb, yb

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
starttime = datetime.now()

print('+--------+-------+----------+----------+-------------+')
print('|  TIME  | ITER  | AVG LOSS | VAL LOSS |  ESTIMATED  |')
print('+--------+-------+----------+----------+-------------+')

for i in range(max_iters):
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if i % training_batches == 0 or i == max_iters-1:
        avgloss = sum(losses)/len(losses)
        losses = []
        
        avloss = '{:08.5f}'.format(avgloss)
        vloss = '{:08.5f}'.format(0 if i == 0 else val_loss())
        
        time = datetime.now()
        s = (time - starttime).total_seconds()
        s = int((max_iters * s) / (i+1e-2))
        h, m = int(s // 3600), int((s % 3600) // 60)
        s = int(s % 60)
        remaining = f'{h:02}:{m:02}:{s:02}'
        print(f'|{time.strftime("%H:%M:%S")}| {i:5} | {avloss} | {vloss} | {(remaining).ljust(12)}|')

print('+--------+-------+----------+----------+-------------+')

#Save the model
print('saving model...')
torch.save(model, 'langgpt/models/model.pt')
