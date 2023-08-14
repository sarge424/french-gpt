import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    '''
    A multihead self-attention class:
        b=batch size
        t=sequence length
        c=encoding size
        
        h=no. of heads
        s=channels per head
        c=h*s
    ---------------------------------
    input = x                    -> (b,t,c)

    k = key(x)                   -> (b,t,c)
    q = query(x)                 -> (b,t,c)
    v = value(x)                 -> (b,t,c)
    
    transform k,q,v              -> (b,t,c) -> (b,t,h,s) -> (b,h,t,s)
    reshape   k,q,v              -> (b,t,h,s) -> (b*h,t,s)
    
    weight = (q * k')/sqrt(c)    -> (b*h,t,t)
    weight = mask(weight)
    weight = softmax(weight)
    
    y = weight * v               -> (b*h,t,s)
    
    reshape   y                  -> (b*h,t,s) -> (b,h,t,s)
    transform y                  -> (b,h,t,s) -> (b,t,h,s) -> (b,t,c)
    
    y = dropout(y)
    y = project(y)               -> (b,t,c) -> (b,t,c)

    output = y                   -> (b,t,c)
    '''
    
    def __init__(self, heads, channels, device):
        super().__init__()
        
        self.h, self.s, self.device = heads, channels // heads, device
        assert channels % heads == 0, \
            f'channels:{channels} not divisible by heads:{heads}'
            
        self.tokey = nn.Linear(channels, channels, bias=False)
        self.toquery = nn.Linear(channels, channels, bias=False)
        self.tovalue = nn.Linear(channels, channels, bias=False)
        
        self.project = nn.Linear(channels, channels)
        
    def forward(self, x):
        b, t, c = x.shape
        h, s = self.h, self.s
        f = b * h
        
        k = self.tokey(x)        # All are b,t,c
        q = self.toquery(x)
        v = self.tovalue(x)
        
        # Convert to f,t,s  (f=b*h)
        k = k.view(b, t, h, s).transpose(1,2).contiguous().view(f, t, s)
        q = q.view(b, t, h, s).transpose(1,2).contiguous().view(f, t, s)
        v = v.view(b, t, h, s).transpose(1,2).contiguous().view(f, t, s)
        
        #output is f,t,t
        w = torch.einsum('fqs,fks->fqk', [q,k]) / (c ** 0.5)
        mask = torch.tril(torch.ones(t, t, dtype=torch.long)).to(self.device)
        w = w.masked_fill(mask == 0, float('-inf'))
        w = F.softmax(w, dim=2)
        
        out = torch.einsum('ftd,fds->fts', [w, v]).view(b, h, t, s)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        
        #project
        out = self.project(out)
        
        return out
        
class TransformerBlock(nn.Module):
    '''
    A reapeatable block with self-attention and feedforward:
        b=batch size
        t=sequence length
        c=encoding size
        g=growth factor
        r=dropout
    --------------------------------------------------------
    feedforward = Linear[g*c->c](ReLU(linear[c->g*c](x)))
    --------------------------------------------------------
    input = x                     -> (b,t,c)
    
    x = x + norm1(selfattn(x))
    out = x + norm2(feedforward(x))
    out = dropout(out)
    
    output = out                  -> (b,t,c)
    '''
    
    def __init__(self, heads, channels, growth, dropout, device):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attention  = SelfAttention(heads, channels, device)
        self.norm2 = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, growth * channels),
            nn.ReLU(),
            nn.Linear(growth * channels, channels),
            
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = x + self.norm1(self.attention(x))
        out = x + self.norm2(self.feedforward(x))
        out = self.dropout(out)
        
        return out
        
class Transformer(nn.Module):
    '''
    A decoder-only transformer with multiple blocks:
        b=block size
        t=sequence length
        c=encoding size
        v=vocab size
        m=max sequence length
        d=no of blocks in sequence
    ------------------------------------------------
    blocks = d ^ TransformerBlock(x)
    logits = Linear[c->v](x)
    wordemb = Linear[v->c](x)
    posemb = Linear[m->c](x)
    ------------------------------------------------
    input = x                                 -> (b,t)
    
    pos = arange(t)                           -> 0,1,...,t-1
    emb = wordemb(x) + expand(posemb(pos))    -> (b,t,c)
    
    emb = blocks(emb)
    logits = tologits(emb)

    output = logits                           -> (b,t,v)
    '''
    
    def __init__(self, heads, channels, growth, dropout, depth, vsize, mlen, device):
        super().__init__()
        self.c, self.m, self.v = channels, mlen, vsize
        self.device = device
        
        blocks = [
            TransformerBlock(heads, channels, growth, dropout, device)
            for _ in range(depth)
        ]
        
        self.wordemb = nn.Embedding(vsize, channels)
        self.posemb = nn.Embedding(mlen, channels)
        
        self.blocks = nn.Sequential(*blocks)
        self.tologits = nn.Linear(channels, vsize)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, x, y=None):
        b, t = x.shape
        c, v = self.c, self.v
        
        p = torch.arange(t, dtype=torch.long).to(self.device)
        pemb = self.posemb(p)[None, :, :].expand(b, t, c)
        emb = self.wordemb(x) + pemb
        
        emb = self.blocks(emb)
        logits = self.tologits(emb)
        
        if y is None:
            loss = None
        else:
            l = logits.view(b * t, v)
            yl = y.view(b * t)
            loss = F.cross_entropy(l, yl)
        
        return logits, loss
    
    def generate(self, x, end):
        while True:
            out, loss = self(x[:, -self.m:])
            ch = torch.multinomial(F.softmax(out[0, -1, :], dim=0), num_samples=1)
            x = torch.cat((x, ch.view(1, 1)), dim=1)
            
            if ch.item() == end:
                break

        return x
    
    def num_params(self):
        return sum(p.numel() for p in self.parameters())