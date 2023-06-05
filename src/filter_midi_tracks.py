from __init__ import *
from data_cleaning import Open, Stream, MakeMIDI, ChordEval
from music21 import chord, note
import os
import numpy as np
import pandas as pd
from pprint import pprint


def get_file_paths():
    test_dir = os.path.join(MIDI_DATA_DIR, "test_data")
    file_paths = []
    for midi_file in os.listdir(test_dir):
        file_paths.append(os.path.join(test_dir, midi_file))

    file_paths = np.array(file_paths)
    return file_paths


def filter_chords(midi_data):
    parts_array = []
    for i, part in enumerate(midi_data.parts):
        chords_array = []
        for element in part.recurse():
            if isinstance(element, chord.Chord):
                chord_array = []
                for i, n in enumerate(element):
                    chord_array.append(
                        np.array([n.pitch, n.volume.velocity, element.offset]))  # type: ignore
                chords_array.append(chord_array)
        if chords_array:
            parts_array.append(chords_array)
    return parts_array


if __name__ == "__main__":
    midi_files = get_file_paths()
    midi_objects = Open.open_midi_files(midi_files)

    filtered_midi_dict = filter_chords(midi_objects[0])
    pprint(filtered_midi_dict)
