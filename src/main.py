from data_cleaning import Open
import os

if __name__ == '__main__':
    mid_file =os.path.join(os.getcwd(), "00001.mid")
    midi_data = Open.open_midi(mid_file)
