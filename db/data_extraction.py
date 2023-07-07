# import __init__
from music21 import chord, note, stream, clef, meter, converter
import os
from tqdm import tqdm
import pandas as pd
import sqlite3


def open_midi(file_path):
    try:
        midi = converter.parse(file_path)
        return midi
    except Exception as e:
        print(f"An error occurred while opening the MIDI file: {e}")

# def get_artist_data()


def filter_data(input_list, prev_index):
    filtered_list = []
    part_start_flag = False

    for element in input_list[prev_index:-1]:
        if element == '<part_start>':
            part_start_flag = True
        elif element == '<meta>' and part_start_flag:
            filtered_list.append('<part_start>')
            filtered_list.append(element)
            part_start_flag = False
        elif part_start_flag is False:
            filtered_list.append(element)

    return filtered_list


def create_data_set(dest_directory):
    mega_midi_set = []
    prev_index = 0
    for root, dirs, files in tqdm(os.walk(dest_directory), desc="Walk Progress "):
        if root == dest_directory:  # Skip the files in the main folder
            continue
        genre = os.path.basename(root)
        for f_i, _file in tqdm(enumerate(files), desc=f"Loaded {genre}: ", leave=False):
            mid_file = os.path.join(root, _file)
            midi_data = open_midi(mid_file)
            mega_midi_set.append(f"<{genre}>")
            file_wo_ext = os.path.splitext(_file)[0]
            query = "SELECT artist_name, song_name FROM dl_urls WHERE song_ID = ?"
            cursor.execute(query, (file_wo_ext,))
            row = cursor.fetchone()
            if row is not None:
                _artist_name, _song_name = row
            else:
                break
            mega_midi_set.append(f"<artist {_artist_name}>")
            # mega_midi_set.append(f"<song {_song_name}>")
            mega_midi_set.append("<song_start>")
            for i, part in enumerate(midi_data.parts):  # type: ignore
                mega_midi_set.append(f"<part_start>")
                for element in part.recurse():
                    # if isinstance(element, meter.TimeSignature):
                    #     mega_midi_set.append(element.ratioString)
                    if isinstance(element, chord.Chord):
                        mega_midi_set.append("<chord_meta>")
                        mega_midi_set.append(
                            f"chord_quarterlength {element.duration.quarterLength.real}")
                        mega_midi_set.append(f"chord_offset {element.offset}")
                        mega_midi_set.append("<chord_start>")
                        for note_i, n in enumerate(element):
                            mega_midi_set.append(
                                f"note_pitch index_{note_i} value_{n.pitch.midi}")  # type: ignore
                            mega_midi_set.append(
                                f"note_velocity index_{note_i} value_{n.volume.velocity}")
                            mega_midi_set.append(
                                f"note_quarterlength index_{note_i} value_{n.duration.quarterLength}")
                            mega_midi_set.append(
                                f"note_offset index_{note_i} value_{n.offset}")

                        mega_midi_set.append("<chord_end>")

            mega_midi_set.append("<song_end>")
            # mega_midi_set[prev_index:-
            #               1] = filter_data(mega_midi_set, prev_index)
            prev_index = len(mega_midi_set) - 1

    return mega_midi_set


folder_path = "/Users/rpbot_mac/Documents/GitHub/MIDI-gen-ai/Ultimate-MIDI-Scraper/data/MIDIdata"
db = "/Users/rpbot_mac/Documents/GitHub/MIDI-gen-ai/Ultimate-MIDI-Scraper/data/db/AllMIDI.sqlite3"

if __name__ == "__main__":

    conn = sqlite3.connect(db)
    cursor = conn.cursor()

    x = create_data_set(folder_path)
    print(x)
