import __init__

import os
import datetime

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import glob

from music21 import chord, note, stream, clef, meter, converter, midi, pitch
import fractions as fract
import re

from tqdm import tqdm

import pandas as pd
import json


MIDI_DATASET_PATH = os.path.join(os.getcwd(), "lib", "MIDI_dataset.parquet")
LATEST_MODEL = os.path.join(
    os.getcwd(), "lib", "pretrained_weights", "latest_weights.pth")
FULL_MIDI_LANG_DATA_LIST = pd.read_parquet(
    MIDI_DATASET_PATH)["full_MIDI_lang_base"].values.tolist()

SAMPLE_MIDI_FILE = os.path.join(
    os.getcwd(), "lib", "sample_midi_files", "New MIDI File.mid")


device = 'cuda' if torch.cuda.is_available() else 'cpu'

unique_chars = sorted(list(set(FULL_MIDI_LANG_DATA_LIST)))

stoi = {ch: i for i, ch in enumerate(unique_chars)}
itos = {i: ch for i, ch in enumerate(unique_chars)}
def encode(s): return [stoi[c] for c in s]
def decode(l): return '\n'.join([itos[i] for i in l])


def create_labels_dict():
    itolabels = {}
    value_pattern = r'value_([\d\.]+)'
    for note_i in itos.keys():
        if note_i in range(0, 465):  # artist names
            itolabels[note_i] = itos[note_i][8:16]+"..."
        elif note_i in range(465, 477):  # special tags
            itolabels[note_i] = itos[note_i]
        elif note_i in range(477, 615):  # chord offset meta data
            itolabels[note_i] = f"offset_{fract.Fraction(itos[note_i].split()[-1])}"
        elif note_i in range(615, 762):  # chord quarterlength meta data
            itolabels[note_i] = f"qlength_{fract.Fraction(itos[note_i].split()[-1])}"
        elif note_i in range(762, 797):
            itolabels[note_i] = itos[note_i]
        elif note_i == 797:
            itolabels[note_i] = "n_offset_0"
        elif note_i in range(798, 923):  # note stuff
            res = re.search(value_pattern, itos[note_i])
            if res:
                itolabels[note_i] = pitch.Pitch(
                    int(res.group(1))).nameWithOctave
        elif note_i == 923:
            itolabels[note_i] = "n_len_1"
        elif note_i in range(924, 1051):
            res = re.search(value_pattern, itos[note_i])
            if res:
                itolabels[note_i] = "velocity "+res.group(1)
    return itolabels


def create_midi_val_dict():
    value_pattern = r'value_([\d\.]+)'
    itomn = {}
    for key in itos.keys():
        if key in range(477, 762):
            itomn[key] = float(fract.Fraction(itos[key].split()[-1]))
        elif key in range(797, 1051):
            res = re.search(value_pattern, itos[key])
            if res:
                try:
                    itomn[key] = int(res.group(1))
                except ValueError:
                    itomn[key] = float(res.group(1))
    return itomn


itomn = create_midi_val_dict()
itolabels = create_labels_dict()

vocab_size = len(unique_chars)
block_size = 128
max_length = 128
batch_size = 32

n_embd = 512
n_heads = 8
forward_expansion = 4  # multiple with n_embd
num_layers = 6
dim_feedforward = n_embd*forward_expansion

dropout = 0.4


MIDI_DATA_TENSOR = torch.tensor(
    encode(FULL_MIDI_LANG_DATA_LIST), dtype=torch.long)

n = int(0.9*len(MIDI_DATA_TENSOR))  # first 90% will be train, rest val
train_data = MIDI_DATA_TENSOR[:n]
val_data = MIDI_DATA_TENSOR[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


def filter_data(input_list, prev_index):
    filtered_list = []
    part_start_flag = False

    for element in input_list[prev_index:-1]:
        if element == '<part_start>':
            part_start_flag = True
        elif element == '<chord_meta>' and part_start_flag:
            filtered_list.append('<part_start>')
            filtered_list.append(element)
            part_start_flag = False
        elif part_start_flag is False:
            filtered_list.append(element)

    return filtered_list


def create_primer_vocab(file_path, genre):
    midi_data = None
    try:
        midi = converter.parse(file_path)
    except Exception as e:
        print(f"An error occurred while opening the MIDI file: {e}")
    else:
        prev_index = 0
        vocab = []
        vocab.append("<song_start>")
        vocab.append(f"<{genre}>")
        if midi_data is not None:
            for i, part in enumerate(midi_data.parts):
                vocab.append(f"<part_start>")
                for element in part.recurse():
                    # if isinstance(element, meter.TimeSignature):
                    #     vocab.append(element.ratioString)
                    if isinstance(element, chord.Chord):
                        vocab.append("<chord_meta>")
                        vocab.append(
                            f"chord_quarterlength {element.duration.quarterLength.real}")
                        vocab.append(
                            f"chord_offset {element.offset}")
                        vocab.append("<chord_start>")
                        for note_i, n in enumerate(element):
                            vocab.append(f"note_{note_i}")
                            vocab.append(
                                f"note_pitch value_{n.pitch.midi}")
                            vocab.append(
                                f"note_velocity value_{n.volume.velocity}")
                            vocab.append(
                                f"note_quarterlength value_{n.duration.quarterLength}")
                            vocab.append(
                                f"note_offset value_{n.offset}")
                        vocab.append("<chord_end>")
            vocab.append("<song_end>")
            vocab[prev_index:-
                  1] = filter_data(vocab, prev_index)
            prev_index = len(vocab)
        return vocab
