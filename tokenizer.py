
class BPEncoder:
    
    def get_corpus(self):
        with open('langgpt/data/data.txt', 'r', encoding='utf-8') as file:
            raw_data = file.read()
        
        corpus = {}
        corp = [line.split() for line in raw_data.splitlines()]
        for line in corp:
            for word in line:
                if word not in corpus:
                    corpus[word] = 1
                else:
                    corpus[word] += 1
                    
        tokens = set(c for c in raw_data if c not in ['\n', ' '])
        tokens = sorted(list(tokens))
        
        return corpus, tokens
            
    def learn(self, iters):
        corpus, tokens = self.get_corpus()
        
        #convert corpus to lists to keep easy tracking
        ckeys, ccounts = [], []
        for k, v in corpus.items():
            ckeys.append('^'.join(ch for ch in k))
            ccounts.append(v)
        
        #tokens is already a list - no need for change
        tcounts = [0 for _ in tokens]
        
        #the ones that need to be updated
        newtokens = [t for t in tokens]
        
        for iter in range(iters):
            #find all token pairs that we should check
            pairs = [(nt, t) for nt in newtokens for t in tokens]
            pairs += [(t, nt) for nt in newtokens for t in tokens]
            
            #create count dict for all pairs
            counts = {}
            parents = {}
            for pair in pairs:
                #print('p', pair)
                p = pair[0] + pair[1]
                if p not in counts:
                    counts[p] = 0
                    parents[p] = pair
            
            #check if each pair is in each word and add counts
            for k, v in counts.items(): 
                for word in corpus:
                    if k in word:
                        counts[k] += corpus[word]
                    
            #add new token
            for k, v in counts.items():
                if k in tokens:
                    tcounts[tokens.index(k)] = v
            
            for k, v in counts.items():
                if k not in tokens and v == max([v for k, v in counts.items() if k not in tokens]):
                    print(f'{iter}: {parents[k][0]} + {parents[k][1]} = {k} --> {v}')
                    tokens.append(k)
                    tcounts.append(v)
                    
                    for i in range(len(ckeys)):
                        ckeys[i].replace(f'{parents[k][0]}-{parents[k][0]}', k)
                    
                    #print('parents', parents[k], counts[k])
                    newtokens = [k, parents[k][0], parents[k][1]]
                    break
                    
        print(len(tokens), 'tokens.')
        for k, v in zip(tokens, tcounts):
            print(k, end=' ')
        print('')
        
        return tokens
        
b = BPEncoder()
t = b.learn(250)

with open('data/tokens.txt', 'w', encoding='utf-8') as file:
    file.write('\n'.join(t))