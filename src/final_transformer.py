from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from music21 import chord, note, stream, clef, meter
from data_cleaning import Open
import os
import numpy as np
import re
import asyncio


batch_size = 4  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)


def open_midi_files(dest_directory):
    all_chords = []
    for root, dirs, files in os.walk(dest_directory):
        # print(dirs)
        for f_i, file in enumerate(files):
            mid_file = os.path.join(root, file)
            midi_data = Open.open_midi(mid_file)
            all_chords.append(f"<song_{f_i}>")
            # prog_array = []
            el_array = []
            for i, part in enumerate(midi_data.parts):  # type: ignore
                all_chords.append(f"<part_{i}>")
                for element in part.recurse():
                    # if isinstance(element, meter.TimeSignature):
                    #     all_chords.append(element.ratioString)
                    if isinstance(element, chord.Chord):
                        all_chords.append("<meta>")
                        all_chords.append(
                            f"chord_quarterlength {element.duration.quarterLength.real}")
                        all_chords.append(f"chord_offset {element.offset}")
                        all_chords.append("<chord_start>")
                        for n in element:
                            all_chords.append(
                                f"note_pitch {n.pitch.midi}")  # type: ignore
                            all_chords.append(
                                f"note_velocity {n.volume.velocity}")
                            all_chords.append(
                                f"note_quarterlength {n.duration.quarterLength}")
                            all_chords.append(f"note_offset {n.offset}")
                        all_chords.append("<end>")

    return all_chords


all_chords = open_midi_files(os.path.join(
    os.getcwd(), "src", "sample_rock_set"))

chars = sorted(list(set(all_chords)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}

# for i in chars:
#     match = re.search(r'(?<=\s)(.*)', i)
#     # print(match)

#     if match:
#         number = match.group()
#         print(eval(number))
#     else:
#         print(i)

itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return '\n'.join([itos[i] for i in l])


data = torch.tensor(encode(all_chords), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        token_embds = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = token_embds + pos_emb
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = BigramLanguageModel()
m = model.to(device)

# logits, loss = m(xb, yb)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)


losses = []


# batch_size = 128
for steps in range(10000):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 1000 == 0:
        losses.append(loss.item())
        print(loss.item())
