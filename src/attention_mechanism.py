import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from music21 import chord, note, stream, clef, meter
from data_cleaning import Open
import os

# GLOBALS

concatenated_array = []
all_chords = []
notes = []
n_to_i = {}
i_to_n = {}
mn_to_nn = {}
vocab_size = 0


def open_midi_files(dest_directory):
    for root, dirs, files in os.walk(dest_directory):
        for file in files:
            mid_file = os.path.join(root, file)
            midi_data = Open.open_midi(mid_file)
            for i, part in enumerate(midi_data.parts):
                for element in part.recurse():
                    if isinstance(element, chord.Chord):
                        concatenated_array.append(element)


print("Opening MIDI files...")
open_midi_files(os.path.join(os.getcwd(), "src", "sample_rock_set"))
print("Opened and stored Data!")

print("Setting up data...")


def setup_data():
    global all_chords, notes, n_to_i, i_to_n, mn_to_nn, vocab_size
    for _chord in concatenated_array:
        chord_arr = []
        for _note in _chord:
            chord_arr.append(_note.pitch.ps)
        all_chords.append(chord_arr)

    flattened_list = [
        int(each_note) for each_chord in all_chords for each_note in each_chord
    ]
    notes = sorted(set(flattened_list))
    n_to_i = {s: i + 1 for i, s in enumerate(notes)}
    n_to_i["."] = 0

    i_to_n = {value: key for key, value in n_to_i.items()}

    mn_to_nn = {n: note.Note(n) for n in range(128)}

    vocab_size = len(i_to_n)


print("Done setting up data...")


def encoder(twod_chord_list):
    return [n_to_i[n] for _chord_list in twod_chord_list for n in _chord_list + ["."]]


def decoder(list_of_keys):
    return [i_to_n[i] for i in list_of_keys]


print("encoding chords")
encoded_chords = torch.tensor(encoder(all_chords), dtype=torch.long)
n = int(0.9*len(encoded_chords))  # first 90% will be train, rest val
train_data = encoded_chords[:n]
val_data = encoded_chords[n:]
print("done encoding chords")


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
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
