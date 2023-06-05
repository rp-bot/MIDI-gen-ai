from music21 import converter
import numpy as np
from __init__ import *
import os


def open_midi(file_path):
    try:
        midi = converter.parse(file_path)
        return midi
    except Exception as e:
        print(f"An error occurred while opening the MIDI file: {e}")


def open_midi_files(file_paths):
    midi_objects = []
    for file_path in file_paths:
        try:
            midi_object = converter.parse(file_path)
            midi_objects.append(midi_object)
        except Exception as e:
            print(f"An error occurred while opening the MIDI file: {e}")
    return midi_objects
