import torch 
from dataclasses import dataclass
import torch.nn as nn 
import torch.nn.functional as F 
from transformers import BertTokenizer

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.linear_1 = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(4 * config.n_embd, config.n_embd)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.gelu(x)
        x = self.linear_2(x)
        return x 
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head 
        self.n_embd = config.n_embd 
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        
    def forward(self, x, mask):
        B, T, C = x.shape
        assert self.n_embd % self.n_head == 0
        n_dim = self.n_embd // self.n_head
        qkv = self.qkv(x)
        
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # B, H, T, C
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
    
    
        res = (q @ k.transpose(-2, -1)) * n_dim**(-0.5) # B, H, T, C @ B, H, C, T 
        res = res.masked_fill(mask == 0, float('-inf'))
        res = F.softmax(res, dim=-1)
        
        res = res @ v # B, H, T, T @ B, H, T, C   
        res = res.contiguous().view(B, T, C)
        return res
        
class Block(nn.Module):
    def __init__(self, config):    
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.mlp = MLP(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x, mask):
        # might be better to layer normalize the entire x before resnet                                
        x = x + self.dropout(self.attn(self.ln1(x), mask.unsqueeze(1).unsqueeze(1))) # B, 1, 1, T
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x 
    

class Sentiment(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wte = nn.Embedding(config.vocab, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        
        self.ln = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd * config.block_size)
        self.ln3 = nn.LayerNorm(config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        
        self.lm_head = nn.Linear(config.n_embd * config.block_size, config.n_embd)
        self.lm_head_1 = nn.Linear(config.n_embd, 2)
        self.config = config
        
        self.blocks = nn.ModuleList([Block(config) for i in range(config.n_layer)])
        
    def forward(self, x, mask):
        wte = self.wte(x)
        positions = torch.arange(config.block_size).to(x.device)
        wpe = self.wpe(positions)
        
        x = self.dropout(wte + wpe) 
        x = self.ln(x)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = x.view(x.size(0), -1)
        
        x = self.ln2(x)
        x = self.lm_head(x)
        x = self.ln3(x) 
        x = self.dropout(x) # fix
        
        x = self.lm_head_1(x)
        return x 


@dataclass 
class EncoderConfig:
#     vocab : int = 50257
    vocab : int = 30720  # bert vocab size 30522
    n_embd : int = 128
    block_size: int = 196
    n_head : int = 8
    n_layer : int = 8
    B : int = 512
    dropout  : int = 0.1 

# global values
encoder_path = 'senti_transformer_model.pt'
config = EncoderConfig()
model = Sentiment(config)
weights = torch.load(encoder_path, weights_only=True, map_location=torch.device('cpu'))
model.load_state_dict(weights)
tkn = BertTokenizer.from_pretrained('bert-base-uncased')

# bert tokenizer
def tokenize(text):
     return tkn.encode_plus(text, padding='max_length', max_length=config.block_size, truncation=True, return_tensors='pt', return_attention_mask=True)

# model prediction
def get_sentiment(text):
    tkn_res = tokenize(text)
    x, mask = tkn_res['input_ids'], tkn_res['attention_mask']
    with torch.no_grad():
        model.eval()
        res = model(x, mask)
        res_t = F.softmax(res, dim=-1)
        res = torch.argmax(res_t, dim=-1)
        res = 'Negative' if res[0] == 0 else 'Positive'
    return res, res_t.reshape(-1).tolist()