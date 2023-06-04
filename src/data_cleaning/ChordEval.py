from music21 import note, chord, stream


class ChordEval:
    def __init__(self, midi_obj):
        self.midi_obj = midi_obj

        self.extracted_dict = self.extract_filter()

    def extract_filter(self):
        """returns a list of tuples with shape 
chords = ([note1, note2,...], time)
notes = (note, time)
        """
        temp_dict = {}
        for i, part in enumerate(self.midi_obj.parts):
            notes = []
            chords = []
            for elem in part.recurse():
                if isinstance(elem, note.Note):
                    notes.append(
                        (elem.pitch, elem.offset, elem.volume.velocity))
                elif isinstance(elem, chord.Chord):
                    for element in elem:
                        chords.append(element.pitch, element.offset,
                                      element.volume.velocity)

            temp_dict[i] = {
                "notes": notes, "chords": chords}

        to_del = []
        for key in temp_dict.keys():
            total = len(temp_dict[key]["chords"]) + \
                len(temp_dict[key]["notes"])
            notes = len(temp_dict[key]["notes"])
            chords = len(temp_dict[key]["chords"])

            Pnotes = notes/total
            Pchords = chords/total

            if Pchords < .95:
                to_del.append(key)

        for key_to_del in to_del:
            del temp_dict[key_to_del]

        return temp_dict
