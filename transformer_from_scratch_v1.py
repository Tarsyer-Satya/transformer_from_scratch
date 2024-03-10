import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size ,heads):
        super().__init__()
        self.embed_size  = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert(
            self.head_dim * self.heads == embed_size
        ),"Embed size is not divisible by heads"


        self.queries = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)
        self.fc_out = nn.Linear(embed_size, embed_size)
        

    def forward(self, keys, queries, values, mask = None):
        N = keys.shape[0]
        key_length, query_length, value_length  = keys.shape[1],  queries.shape[1],  values.shape[1]

        keys = self.keys(keys)
        values = self.values(values)
        queries = self.values(queries)  


        queries = queries.view(N, query_length, self.heads, self.head_dim).permute(0,2,1,3)
        values = values.view(N, value_length, self.heads, self.head_dim).permute(0,2,1,3)
        keys = keys.view(N, key_length, self.heads, self.head_dim).permute(0,2,1,3)


        attention_scores = torch.matmul(queries, keys.permute(0,1,3,2))/ torch.sqrt(torch.tensor(self.head_dim, dtype = torch.float32))

        if mask is not None: 
            attention_scores = attention_scores.masked_fill(mask == 0, float("-1e20"))
        
        attention_weights = torch.softmax(attention_scores, dim = -1)

        attention_output = torch.matmul(attention_weights, values)

        attention_output = attention_output.permute(0,2,1,3).contiguous().view(N, query_length, self.embed_size)

        out = self.fc_out(attention_output)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self,embed_size,dropout,forward_expansion,heads):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, embed_size*forward_expansion),
            nn.ReLU(),
            nn.Linear(embed_size*forward_expansion, embed_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, value, key, mask):
        attention = self.attention(key, query, value, mask)
        x = self.dropout(self.norm1(query + attention))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
class Encoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            dropout,
            forward_expansion,
            heads,
            num_layers,
            max_length,
            device
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(trg_vocab_size,embed_size)
        self.positional_embeddings = nn.Embedding(max_length, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(embed_size,dropout,forward_expansion,heads)
            for _ in range(num_layers)
        ])
        self.device = device

    def forward(self,x,mask = None):
        N, seq_length = x.shape


        positions = torch.arange(0, seq_length).expand(N,seq_length).to(self.device)

        out = self.dropout(self.word_embeddings(x) + self.positional_embeddings(positions))

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out
    
class DecoderBlock(nn.Module):
    def __init__(self, embed_size,dropout,forward_expansion,heads):
        super().__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads)
        self.transformer_block = TransformerBlock(embed_size,dropout,forward_expansion,heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x, value, key, src_mask=None, trg_mask=None):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(query, value, key, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
            self,
            trg_vocab_size,
            embed_size,
            max_length, 
            dropout,
            forward_expansion,
            num_layers,
            heads,
            device
    ):
        super().__init__()
        self.word_embeddings = nn.Embedding(trg_vocab_size, embed_size)
        self.positional_embeddings = nn.Embedding(max_length,embed_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, dropout, forward_expansion, heads)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.device = device

    def forward(self, x, enc_out, src_mask = None, trg_mask = None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout(self.word_embeddings(x) + self.positional_embeddings(positions))
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)
        out = self.fc_out(x)
        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_mask_ind,
        trg_mask_ind,
        src_vocab_size,
        trg_vocab_size,
        device = "cpu",
        embed_size = 512,
        dropout = 0.1,
        forward_expansion = 4,
        heads = 8,
        num_layers = 6,
        max_length = 30

    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            dropout,
            forward_expansion,
            heads,
            num_layers,
            max_length,
            device = device
        )
        self.decoder = Decoder(
            trg_vocab_size,
            embed_size,
            max_length, 
            dropout,
            forward_expansion,
            num_layers,
            heads,
            device = device
        )
    
        self.src_pad_ind = src_mask_ind
        self.trg_pad_ind = trg_mask_ind
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_ind).unsqueeze(1).unsqueeze(2).bool().to(self.device)
        return src_mask
    def make_trg_mask(self, trg):
        N, trg_length = trg.shape
        trg_mask = torch.tril(torch.ones((trg_length, trg_length))).expand(
            N,1,trg_length, trg_length
        ).bool()
        tgt_pad_mask = (trg.cpu() != self.trg_pad_ind).unsqueeze(1).unsqueeze(2).bool()
        tgt_mask = trg_mask & tgt_pad_mask
        return trg_mask.to(self.device)
    
    def forward(self,src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src,src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return out

    

# def test():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(device)

#     src_mask_ind = 0
#     trg_mask_ind = 0
#     src_vocab_size = 1000
#     trg_vocab_size = 1000

#     transformer = Transformer(
#         src_mask_ind=src_mask_ind,
#         trg_mask_ind=trg_mask_ind,
#         src_vocab_size=src_vocab_size,
#         trg_vocab_size=trg_vocab_size,
#         device = device
#     )

#     src = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 1], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(device)
#     trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 1], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

#     out = transformer(src, trg)
#     print(out.shape)


# test()




    
