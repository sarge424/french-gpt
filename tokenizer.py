import json

class Token:
    def __init__(self, c, parents=None):
        self.ch = c
        self.parents = parents
        if self.parents is None:
            self.chs = ' '+ self.ch + ' '
        else:
            self.chs = ' ' + ' '.join(self.parents) + ' '
        
    def __add__(self, other):
        assert isinstance(other, Token), f'cant merge token and {type(other)}'
        return Token(self.ch + other.ch, (self.ch, other.ch))

    def __repr__(self):
        return f'{self.ch}'

    def __eq__(self, other):
        return self.ch == other.ch

    def __hash__(self):
        return hash((self.ch, self.parents))

    def count_in(self, corpus, ccounts):
        count = 0
        for word, wcount in zip(corpus, ccounts):
            c = word.count(self.chs)
            count += wcount * c
        return count

    def to_json(self):
        return {'ch': self.ch, 'parents': self.parents}

    def from_json(d):
        return Token(d['ch'], d['parents'])

class BPEncoder:
    
    def __init__(self):
        self.tokens = []
     
    def load_tokens(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            j = file.read()
            
        x = json.loads(j)
        self.tokens = [Token.from_json(d) for d in x]
        
    def save_tokens(self, filepath):
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump([obj.to_json() for obj in self.tokens], file)
        
    def use_corpus(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            raw_data = file.read()
        
        #store a dict with word: count for each unique word in the dataset
        corpus = {}
        corp = [line.split() for line in raw_data.splitlines()]
        for line in corp:
            for word in line:
                if word not in corpus:
                    corpus[word] = 1
                else:
                    corpus[word] += 1
                    
        #tokens is a sorted list of Tokens of 1 char length (excluding spaces and special tokens)
        tokens = set(c for c in raw_data if c not in ['\n', ' ', '/'])
        tokens = sorted(list(tokens))
        self.tokens = [Token(t) for t in tokens]
        
        return corpus
            
    def learn(self, inp_file, tlen):
        corpusdict = self.use_corpus(inp_file)
        
        #split the corpus into two lists for easier access.
        #each word is split with spaces to make token searching simpler
        corpus, ccounts = [], []
        for k, v in corpusdict.items():
            corpus.append(' ' + ' '.join(c for c in k) + ' ')
            ccounts.append(v)
        
        #list of tokens that need their counts (and counts of their children) to be updated
        newtokens = [t for t in self.tokens]
        
        counts = {t: t.count_in(corpus, ccounts) for t in self.tokens} #count of each token
        used = {t: True for t in self.tokens}                          #is this token already in vocab?
        
        print('init:', len(self.tokens), 'chars found.')
        
        while len(self.tokens) < tlen-3: #-3 to include space for specials
            #create next batch of tokens from newly created tokens
            #if e + r -> er, we only want to update counts for tokens made up of e, r and er
            #since the counts are added to the global count pool, they will be compared 
            #againts previously calculated counts for all other tokens
            pairs = [nt for nt in newtokens]
            pairs = [nt + t for nt in newtokens for t in self.tokens]
            pairs += [t + nt for nt in newtokens for t in self.tokens if (t+nt) not in pairs]
            
            #find the count for newly made tokens and add them to the count pool
            for t in pairs:
                tc = t.count_in(corpus, ccounts)
                counts[t] = tc
                used[t] = False
            
            #add the best token to the vocab
            best, bcount = None, -1
            for t, c in counts.items():
                if not used[t] and c > bcount:
                    best, bcount = t, c
                    
            self.tokens.append(best)
            
            #tokens that need to be updated next iteration
            newtokens = [best] + [Token(p) for p in best.parents]
            
            print(len(self.tokens), best.parents, best, bcount)
            
            #update the words in the corpus to include the new token.
            #' s i m p l e r _ ' -> merge(e, r) -> ' s i m p l er _ '
            for i in range(len(corpus)):
                corpus[i] = corpus[i].replace(best.chs, f' {best.ch} ')
                
        #add special tokens
        self.tokens = [Token('*'), Token('^'), Token('/')] + self.tokens
        
    def tokenize_file(self, inp_file, out_file):
        with open(inp_file, 'r', encoding='utf-8') as file:
            data = [line.split() for line in file.read().splitlines()]
        
        data_t = []
        for i in range(len(data)):
            line_t = []
            for j in range(len(data[i])):
                line_t.extend(self.tokenize(data[i][j]))
            data_t.append(' '.join(line_t))
        
        with open(out_file, 'w', encoding='utf-8') as file:
            file.write('\n'.join(data_t))
    
    def tokenize(self, word):
        word = f' {" ".join(c for c in word)} '
        for token in self.tokens:
            if token.chs in word:
                word = word.replace(token.chs, f' {token.ch} ')
        word = word.split()
        return word

    def get_functions(self):
        stoi = {t.ch: i for i,t in enumerate(self.tokens)}
        itos = {i: t.ch for i,t in enumerate(self.tokens)}
        
        return stoi, itos
    
    def encode(self, data):
        stoi, _ = self.get_functions()
            
        return [[stoi[t] for t in line] for line in data]
    
    def decode(self, data):
        _, itos = self.get_functions()
            
        return [[itos[t] for t in line] for line in data]