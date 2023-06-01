from music21 import converter
from __init__ import *
import os


def open_midi(file_path):
    try:
        midi = converter.parse(file_path)
        return midi
    except Exception as e:
        print(f"An error occurred while opening the MIDI file: {e}")