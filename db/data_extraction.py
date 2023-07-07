import __init__
from music21 import chord, note, stream, clef, meter, converter
import os
from tqdm import tqdm
import pandas as pd
import sqlite3
import numpy as np

MIDI_DATA_DIR = os.path.join(
    os.getcwd(), "Ultimate-MIDI-Scraper", "data", "MIDIdata")

FOLDER_PATH = os.path.join(
    os.getcwd(), "Ultimate-MIDI-Scraper", "data", "MIDIdata", "classic")

DB = os.path.join(
    os.getcwd(), "Ultimate-MIDI-Scraper", "data", "db", "AllMIDI.sqlite3")


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
        elif element == '<chord_meta>' and part_start_flag:
            filtered_list.append('<part_start>')
            filtered_list.append(element)
            part_start_flag = False
        elif part_start_flag is False:
            filtered_list.append(element)

    return filtered_list


def determine_max_num_songs(genre_dir, percentile):
    # List all folders
    if MIDI_DATA_DIR is not None:
        folders = [os.path.join(MIDI_DATA_DIR, f) for f in os.listdir(
            MIDI_DATA_DIR) if os.path.isdir(os.path.join(MIDI_DATA_DIR, f))]

        # Count the number of files in each folder
        file_counts = [len([f for f in os.listdir(folder) if os.path.isfile(
            os.path.join(folder, f))]) for folder in folders]
    else:
        file_counts = len(os.listdir(genre_dir))
    # print([os.path.basename(folder) for folder in folders], file_counts)
    # Calculate the specified percentile
    return int(np.percentile(file_counts, percentile))


def determine_rand_range(sub_dir, max_count):
    if len(os.listdir(sub_dir)) <= max_count:
        max_count = len(os.listdir(sub_dir))

    random_indices = np.random.randint(
        low=0, high=max_count, size=max_count).tolist()
    return random_indices


def create_data_set(dest_directory: str):
    mega_midi_set = []
    prev_index = 0

    max_num_songs = determine_max_num_songs(
        genre_dir=dest_directory, percentile=28)

    for root, dirs, files in tqdm(os.walk(dest_directory), desc="Walk Progress "):
        # if root == dest_directory:  # Skip the files in the main folder
        #     continue
        genre = os.path.basename(root)

        rand_indices = determine_rand_range(root, max_num_songs)
        new_files_list = [files[rand_i] for rand_i in rand_indices]

        for f_i, _file in tqdm(enumerate(new_files_list), desc=f"Loaded {genre}: ", leave=False):
            mid_file = os.path.join(root, _file)
            midi_data = open_midi(mid_file)
            mega_midi_set.append("<song_start>")
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
            if midi_data is not None:
                for i, part in enumerate(midi_data.parts):
                    mega_midi_set.append(f"<part_start>")
                    for element in part.recurse():
                        # if isinstance(element, meter.TimeSignature):
                        #     mega_midi_set.append(element.ratioString)
                        if isinstance(element, chord.Chord):
                            mega_midi_set.append("<chord_meta>")
                            mega_midi_set.append(
                                f"chord_quarterlength {element.duration.quarterLength.real}")
                            mega_midi_set.append(
                                f"chord_offset {element.offset}")
                            mega_midi_set.append("<chord_start>")
                            for note_i, n in enumerate(element):
                                mega_midi_set.append(f"note_{note_i}")
                                mega_midi_set.append(
                                    f"note_pitch value_{n.pitch.midi}")
                                mega_midi_set.append(
                                    f"note_velocity value_{n.volume.velocity}")
                                mega_midi_set.append(
                                    f"note_quarterlength value_{n.duration.quarterLength}")
                                mega_midi_set.append(
                                    f"note_offset value_{n.offset}")

                            mega_midi_set.append("<chord_end>")

                mega_midi_set.append("<song_end>")
                mega_midi_set[prev_index:-
                              1] = filter_data(mega_midi_set, prev_index)
                prev_index = len(mega_midi_set)

    return mega_midi_set


if __name__ == "__main__":

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    # original_list = pd.read_parquet("large_deconstructed_midi.parquet")['Giga_MIDI_Language'].values.tolist()
    large_MIDI_language_list = create_data_set(FOLDER_PATH)

    cursor.close()
    conn.close()

    df = pd.DataFrame(large_MIDI_language_list, columns=['classical'])
    df.to_parquet('classical_deconstructed_midi.parquet')
