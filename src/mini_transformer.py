import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from music21 import chord, note, stream, clef, meter
from data_cleaning import Open
import os
import numpy as np


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
files_array = []


def open_midi_files(dest_directory):
    for root, dirs, files in os.walk(dest_directory):
        for file in files:
            mid_file = os.path.join(root, file)
            midi_data = Open.open_midi(mid_file)

            prog_array = []
            for i, part in enumerate(midi_data.parts):  # type: ignore
                el_array = []
                for element in part.recurse():
                    if isinstance(element, chord.Chord):
                        el_array.append([p.midi for p in element.pitches])
                if el_array:
                    prog_array.append(el_array)

            files_array.append(prog_array)


open_midi_files(os.path.join(os.getcwd(), "src", "sample_rock_set"))

flattened_list = [
    each_note
    for each_file in files_array
    for each_prog in each_file
    for each_chord in each_prog
    for each_note in each_chord
]
notes = sorted(set(flattened_list))
n_to_i = {s: i for i, s in enumerate(notes)}
n_to_i[128] = len(n_to_i)

i_to_n = {value: key for key, value in n_to_i.items()}

mn_to_nn = {n: note.Note(n) for n in range(128)}

vocab_size = len(i_to_n)


def paddify():
    max_num_progressions = max([len(x) for x in files_array])
    max_num_chords = max([len(_prog) for _f in files_array for _prog in _f])
    max_num_notes = max(
        [len(_chord) for _f in files_array for _prog in _f for _chord in _prog]
    )
    padded_files_array = []
    for file in files_array:
        temp_file = file + [[[128]]] * (max_num_progressions - len(file))
        padded_progressions_array = []

        for i, progression in enumerate(temp_file):
            padded_chords_array = []
            temp_progression = progression + \
                [[128]] * (max_num_chords - len(progression))

            for _chord in temp_progression:
                temp_chord = _chord + [128] * (max_num_notes - len(_chord))
                padded_chords_array.append([n_to_i[n] for n in temp_chord])
            padded_progressions_array.append(padded_chords_array)

        padded_files_array.append(padded_progressions_array)

    return torch.tensor(padded_files_array).view(98, 551, 8).float()


_data = paddify()

n = int(0.9 * _data.shape[1])  # first 90% will be train, rest val
train_data = _data[:, :n]
val_data = _data[:, n:]




# model = GPTLanguageModel()
# m = model.to(device)
# # print the number of parameters in the model
# print(sum(p.numel() for p in m.parameters()), 'M parameters')

# # create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))
