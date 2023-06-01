"""make a midi file using this class"""
from music21 import chord, stream


class MakeMIDI:
    def __init__(self, midi_obj):
        self.midi_obj = midi_obj
        self.modified_midi = self.get_chord_tracks()

    def get_chord_tracks(self):
        try:
            # Verify the input is a Score object
            if not isinstance(self.midi_obj, stream.Score):
                raise TypeError("Input must be a Score object.")

        # Container for tracks with chords
            tracks_with_chords = stream.Score()

        # Check each part for chords
            for part in self.midi_obj.parts:
                if any(isinstance(elem, chord.Chord) for elem in part.recurse()):
                    tracks_with_chords.insert(0, part)

            return tracks_with_chords

        except Exception as e:
            print(f"An error occurred: {e}")

    def create_file(self, filename="filetered_file.mid"):
        self.modified_midi.write('midi', filename)
