import mido
import time
from music21 import note, stream

buffer = 0.1

# List to hold the current group of notes and the final groups
current_group = []
groups = []

# Record the start time
start_time = time.time()
i = 0
with mido.open_input('iRig KEYS 37 1') as inport:
    while i < 10:
        for msg in inport:
            if msg.type == 'note_on':
                # If the time since the last note_on message is larger than the buffer,
                # we assume that the previous group of notes is finished
                current_group.append([msg.note, msg.velocity])
                if time.time() - start_time > buffer and len(current_group) > 1:
                    groups.append(current_group)
                    print("<chord_start>")
                    for n_i, n_v in enumerate(current_group):
                        print(f"note_{n_i}",f"notenote_pitch value_{n_v[0]}", f"note_velocity value_{n_v[1]}")
                    print("<chord_end>")
                    current_group = []
                    i += 1

                # Append the note and velocity to the current group
                

                # Update the start time
                start_time = time.time()

# Add the final group if it's not empty
if current_group:
    groups.append(current_group)
