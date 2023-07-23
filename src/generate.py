# -*- coding: utf-8 -*-
from model_15M import *


# @param ["classic", "country", "pop", "rap_hip_hop", "rnb_soul", "rock"]


# @markdown So an approximate ratio would be 20 tokens to 1 chord
# @param {type:"slider", min:40, max:1000, step:10}


def filter_data_and_organize(input_list):
    sublists = []
    sublist = []
    tags_dict = {}
    for i, element in enumerate(input_list):
        # if the element starts with "<" and ends with ">"
        if element in range(0, 477):
            # if the current sublist is not empty, append it to the sublists
            if sublist:
                tags_dict[len(sublists)-1] = element
                sublists.append(sublist)

                sublist = []
        else:
            sublist.append(element)
    if sublist:
        sublists.append(sublist)

    # Join META DATA with lists
    chord_meta_index = None
    combined_list = []
    for i, chrd_list in enumerate(sublists):
        if any(477 <= num < 762 for num in chrd_list) and not any(762 <= num < 1051 for num in chrd_list):
            if chord_meta_index is not None:
                chord_meta_index = None
            else:
                chord_meta_index = i
        elif any(762 <= num < 1051 for num in chrd_list) and not any(477 <= num < 762 for num in chrd_list) and chord_meta_index is not None:
            if chord_meta_index+1 == i:
                combined_list.append(
                    sublists[chord_meta_index]+sublists[i])
                chord_meta_index = None

        elif chord_meta_index is None:
            combined_list.append(chrd_list)

    # create_notes_lists within the chords list
    note_index_pattern = r'note_(\d+)'
    final_list = []

    for input_list in combined_list:
        output_list = []
        temp_list = []
        chord_list = []
        for i in input_list:
            if i in range(477, 762):
                chord_list.append(i)
            elif i in range(762, 797):
                if temp_list:
                    output_list.append(temp_list)
                    temp_list = []
            else:
                temp_list.append(i)

        # Append the final sublist and chord list if they're not empty
        if temp_list:
            output_list.append(temp_list)
        if chord_list:
            output_list.append(chord_list)

        final_list.append(output_list)

    return final_list

def create_midi_file(input_list, output_file_name):
    s = stream.Stream()
    for list_of_chords in input_list:
        c = chord.Chord()
        c_meta_data = {}
        notes_list = []
        for i, note_data in enumerate(list_of_chords):

            if any(477 <= num < 762 for num in note_data):
                for note_item in note_data:
                    if note_item in range(477, 615):
                        c.offset = itomn[note_item]
                        pass
                    else:
                        c.duration.quarterLength = itomn[note_item]
                        pass
            else:
                n = note.Note()
                for note_item in note_data:
                    if any(798 <= num < 923 for num in note_data):
                        if note_item in range(798, 923):
                            n.pitch.midi = itomn[note_item]
                        if note_item == 797:
                            n.offset = 0.0
                        elif note_item == 923:
                            n.duration.quarterLength = 1.0
                        elif note_item in range(924, 1051):
                            n.volume.velocity = itomn[note_item]

                notes_list.append(n)
        c.add(notes_list)
        s.append(c)
    s.write('midi', fp=f'{output_file_name}.mid')



if __name__ == '__main__':
    model_gen = MIDIGenModel()
    if device == "cuda":
        model_gen.load_state_dict(torch.load(LATEST_MODEL))
    else:
        model_gen.load_state_dict(torch.load(
            LATEST_MODEL, map_location=torch.device('cpu')))
    model_gen = model_gen.to(device)

    genre = "rock"

    primer_vocab = create_primer_vocab(SAMPLE_MIDI_FILE, genre)
    primer = [stoi[midi_word] for midi_word in primer_vocab]
    context = torch.tensor([primer], dtype=torch.long, device=device)

    chord_progression_length = 50
    p_amount = 0.90  
    generated_out1, probs_1 = model_gen.generate(
    context, chord_progression_length, p_amount)
    gen_list = generated_out1[0].cpu().detach().numpy().tolist()

    sjdh = [itolabels[i] for i in gen_list]
    chords_out = filter_data_and_organize(gen_list)
    
    create_midi_file(chords_out, "sample1")