# init file for folder level 2
import os
import sys
# the deeper you go, the more times you'll have to call this function
currentdir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

path_to_dir = ["Ultimate-MIDI-Scraper", "data", "MIDIdata"]
MIDI_DATA_DIR = os.path.join(parentdir, *path_to_dir)
