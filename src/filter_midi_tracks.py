from __init__ import *
from data_cleaning import Open, Stream, MakeMIDI, ChordEval
from music21 import chord, note, stream
import os
import numpy as np
import pandas as pd
from pprint import pprint
from tqdm import tqdm


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
                    chord_array.append(n)
                chords_array.append(chord_array)
        if chords_array:
            parts_array.append(chords_array)
    return parts_array


def write_midi_files(midi_data):
    streams = []
    for track in tqdm(midi_data):

        for chord in track:
            s = stream.Stream()
            for pitch, velocity, offset in chord:
                # Create a note with the given pitch
                n = note.Note(midi=pitch.ps)
                n.volume.velocity = velocity  # Set the velocity
                n.offset = offset  # Set the offset
                s.append(n)  # Add the note to the stream
            streams.append(s)

    # Create a Score and add each stream as a part
    score = stream.Score()
    for s in streams:
        part = stream.Part(s)
        score.insert(0, part)

    # Write the Score to a MIDI file
    score.write('midi', 'output_file.mid')


if __name__ == "__main__":
    midi_files = get_file_paths()
    midi_objects = Open.open_midi_files(midi_files)

    filtered_midi_dict = filter_chords(midi_objects[0])

    # write_midi_files(filtered_midi_dict)

    x = filtered_midi_dict[1]  # [part][chord][note]
    pprint(filtered_midi_dict, indent=2)
