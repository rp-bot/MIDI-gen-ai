# init file for folder level 1
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

path_to_dir = ["Ultimate-MIDI-Scraper", "data", "MIDIdata"]
DATA_DIR = os.path.join(os.path.abspath(''), *path_to_dir)
