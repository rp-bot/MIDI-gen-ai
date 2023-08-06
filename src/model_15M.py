import __init__
from lib.model_setup.environment_setup import *

class SelfAttention(nn.Module):
    def __init__(self):
        super(SelfAttention, self).__init__()
        self.embed_size = n_embd
        self.heads = n_heads
        self.head_dim = n_embd // n_heads

        assert (
            self.head_dim * n_heads == n_embd
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(n_heads * self.head_dim, n_embd)

    def forward(self, values, keys, query, mask=None):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        # values, keys, queries: (N, S, H, E/H)        (32, 20, 8, 8)


        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # energy: (N, H, S, S)                         (32, 8, 20, 20)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))



        attention = torch.softmax(energy / (self.embed_size ** 0.5), dim=3)
        # attention: (N, H, S, S)                      (32, 8, 20, 20)


        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # out: (N, S, E)    (32, 20, 64)


        out = self.fc_out(out)
        # out: (N, S, E)    (32, 20, 64)
        return out

#@title Decoderblock Class
class DecoderBlock(nn.Module):
    def __init__(self):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention()
        self.norm1 = nn.LayerNorm(n_embd)
        self.norm2 = nn.LayerNorm(n_embd)

        self.feed_forward = nn.Sequential(
            nn.Linear(n_embd, forward_expansion * n_embd),
            # self.feed_forward[0](x): (N, S, E*F)  (32, 20, 2048)
            nn.ReLU(),
            # self.feed_forward[2](self.feed_forward[0](x)): (32, 20, 64)
            nn.Linear(forward_expansion * n_embd, n_embd),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask=None):
        # query, key, value: (N, S, E)       (32, 20, 64)

        attention = self.attention(value, key, query, mask)
        # attention: (N, S, E)               (32, 20, 64)

        x = self.dropout(self.norm1(attention + query))
        # x: (N, S, E)                       (32, 20, 64)

        forward = self.feed_forward(x) # -> runs a sequential class
        # forward: (N, S, E)                 (32, 20, 64)

        out = self.dropout(self.norm2(forward + x))
        # out: (N, S, E)                     (32, 20, 64)
        return out

#@title MIDI Chord Gen Class
class MIDIGenModel(nn.Module):
    def __init__(self):
        super(MIDIGenModel, self).__init__()

        self.embed_size = n_embd
        self.device = device

        self.word_embedding = nn.Embedding(vocab_size, n_embd)
        # self.word_embedding(x): (N, S, E)             (32, 20, 64)

        self.position_embedding = nn.Embedding(max_length, n_embd)
        # self.position_embedding(positions): (N, S, E) (32, 20, 64)


        self.layers = nn.ModuleList(
            [
                DecoderBlock()
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(n_embd, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, targets=None):
        # x: (N, S)                                     (32, 20)
        N, seq_length = x.shape


        positions = (
            torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        )

        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )
        # out: (N, S, E)                                (32, 20, 64)
        mask = torch.tril(torch.ones((seq_length, seq_length))
               .type(torch.BoolTensor)
               ).to(self.device)

        for layer in self.layers:
            # each transformer block
            out = layer(out, out, out, mask) # out: (N, S, E) (32, 20, 64)

        out = self.fc_out(out) # (32, 20, 64)

        if targets is None:
            loss = None

        else:
            B, T, C = out.shape
            logits = out.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return out, loss

    def generate(self, idx, max_new_tokens, p=0.9):
        probs_list=[]
    # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)

            # Apply top-p nucleus sampling
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[:, indices_to_remove] = float('-inf')

            # Sample from the remaining distribution
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            probs_list.append(probs.cpu().detach().numpy())
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx, probs_list
