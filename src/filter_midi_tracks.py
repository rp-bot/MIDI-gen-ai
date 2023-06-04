from __init__ import *
from data_cleaning import Open, Stream, MakeMIDI, ChordEval
import os


def main():
    pass

# mid_file = os.path.join(os.getcwd(), ["0001.mid"])
# midi_data = Open.open_midi(mid_file)


if __name__ == "__main__":
    print(os.listdir(TEST_DATA_DIR))
