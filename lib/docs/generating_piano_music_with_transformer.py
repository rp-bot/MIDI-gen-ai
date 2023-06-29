print('Copying Salamander piano SoundFont (via https://sites.google.com/site/soundfonts4u) from GCS...')
# !gsutil -q -m cp -r gs://magentadata/models/music_transformer/primers/* /content/
# !gsutil -q -m cp gs://magentadata/soundfonts/Yamaha-C5-Salamander-JNv5.1.sf2 /content/

print('Installing dependencies...')
# !apt-get update -qq && apt-get install -qq libfluidsynth2 build-essential libasound2-dev libjack-dev
# !pip install -q 'tensorflow-datasets < 4.0.0'
# !pip install -qU google-cloud magenta pyfluidsynth

print('Importing libraries...')

import numpy as np
import os
import tensorflow.compat.v1 as tf

from google.colab import files

from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf
import note_seq

tf.disable_v2_behavior()

print('Done!')


SF2_PATH = '/content/Yamaha-C5-Salamander-JNv5.1.sf2'
SAMPLE_RATE = 16000

# Upload a MIDI file and convert to NoteSequence.
def upload_midi():
  data = list(files.upload().values())
  if len(data) > 1:
    print('Multiple files uploaded; using only one.')
  return note_seq.midi_to_note_sequence(data[0])

# Decode a list of IDs.
def decode(ids, encoder):
  ids = list(ids)
  if text_encoder.EOS_ID in ids:
    ids = ids[:ids.index(text_encoder.EOS_ID)]
  return encoder.decode(ids)

"""# Piano Performance Language Model"""

#@title Setup and Load Checkpoint
#@markdown Set up generation from an unconditional Transformer
#@markdown model.

model_name = 'transformer'
hparams_set = 'transformer_tpu'
ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/unconditional_model_16.ckpt'

class PianoPerformanceLanguageModelProblem(score2perf.Score2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = PianoPerformanceLanguageModelProblem()
unconditional_encoders = problem.get_feature_encoders()

# Set up HParams.
hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
trainer_lib.add_problem_hparams(hparams, problem)
hparams.num_hidden_layers = 16
hparams.sampling_method = 'random'

# Set up decoding HParams.
decode_hparams = decoding.decode_hparams()
decode_hparams.alpha = 0.0
decode_hparams.beam_size = 1

# Create Estimator.
run_config = trainer_lib.create_run_config(hparams)
estimator = trainer_lib.create_estimator(
    model_name, hparams, run_config,
    decode_hparams=decode_hparams)

# Create input generator (so we can adjust priming and
# decode length on the fly).
def input_generator():
  global targets
  global decode_length
  while True:
    yield {
        'targets': np.array([targets], dtype=np.int32),
        'decode_length': np.array(decode_length, dtype=np.int32)
    }

# These values will be changed by subsequent cells.
targets = []
decode_length = 0

# Start the Estimator, loading from the specified checkpoint.
input_fn = decoding.make_input_fn_from_generator(input_generator())
unconditional_samples = estimator.predict(
    input_fn, checkpoint_path=ckpt_path)

# "Burn" one.
_ = next(unconditional_samples)

#@title Generate from Scratch
#@markdown Generate a piano performance from scratch.
#@markdown
#@markdown This can take a minute or so depending on the length
#@markdown of the performance the model ends up generating.
#@markdown Because we use a
#@markdown [representation](http://g.co/magenta/performance-rnn)
#@markdown where each event corresponds to a variable amount of
#@markdown time, the actual number of seconds generated may vary.

targets = []
decode_length = 1024

# Generate sample events.
sample_ids = next(unconditional_samples)['outputs']

# Decode to NoteSequence.
midi_filename = decode(
    sample_ids,
    encoder=unconditional_encoders['targets'])
unconditional_ns = note_seq.midi_file_to_note_sequence(midi_filename)

# Play and plot.
note_seq.play_sequence(
    unconditional_ns,
    synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(unconditional_ns)

#@title Download Performance as MIDI
#@markdown Download generated performance as MIDI (optional).

note_seq.sequence_proto_to_midi_file(
    unconditional_ns, '/tmp/unconditional.mid')
files.download('/tmp/unconditional.mid')

#@title Choose Priming Sequence
#@markdown Here you can choose a priming sequence to be continued
#@markdown by the model.  We have provided a few, or you can
#@markdown upload your own MIDI file.
#@markdown
#@markdown Set `max_primer_seconds` below to trim the primer to a
#@markdown fixed number of seconds (this will have no effect if
#@markdown the primer is already shorter than `max_primer_seconds`).

filenames = {
    'C major arpeggio': '/content/c_major_arpeggio.mid',
    'C major scale': '/content/c_major_scale.mid',
    'Clair de Lune': '/content/clair_de_lune.mid',
}
primer = 'C major scale'  #@param ['C major arpeggio', 'C major scale', 'Clair de Lune', 'Upload your own!']

if primer == 'Upload your own!':
  primer_ns = upload_midi()
else:
  # Use one of the provided primers.
  primer_ns = note_seq.midi_file_to_note_sequence(filenames[primer])

# Handle sustain pedal in the primer.
primer_ns = note_seq.apply_sustain_control_changes(primer_ns)

# Trim to desired number of seconds.
max_primer_seconds = 20  #@param {type:"slider", min:1, max:120}
if primer_ns.total_time > max_primer_seconds:
  print('Primer is longer than %d seconds, truncating.' % max_primer_seconds)
  primer_ns = note_seq.extract_subsequence(
      primer_ns, 0, max_primer_seconds)

# Remove drums from primer if present.
if any(note.is_drum for note in primer_ns.notes):
  print('Primer contains drums; they will be removed.')
  notes = [note for note in primer_ns.notes if not note.is_drum]
  del primer_ns.notes[:]
  primer_ns.notes.extend(notes)

# Set primer instrument and program.
for note in primer_ns.notes:
  note.instrument = 1
  note.program = 0

# Play and plot the primer.
note_seq.play_sequence(
    primer_ns,
    synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(primer_ns)

#@title Generate Continuation
#@markdown Continue a piano performance, starting with the
#@markdown chosen priming sequence.

targets = unconditional_encoders['targets'].encode_note_sequence(
    primer_ns)

# Remove the end token from the encoded primer.
targets = targets[:-1]

decode_length = max(0, 4096 - len(targets))
if len(targets) >= 4096:
  print('Primer has more events than maximum sequence length; nothing will be generated.')

# Generate sample events.
sample_ids = next(unconditional_samples)['outputs']

# Decode to NoteSequence.
midi_filename = decode(
    sample_ids,
    encoder=unconditional_encoders['targets'])
ns = note_seq.midi_file_to_note_sequence(midi_filename)

# Append continuation to primer.
continuation_ns = note_seq.concatenate_sequences([primer_ns, ns])

# Play and plot.
note_seq.play_sequence(
    continuation_ns,
    synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(continuation_ns)

#@title Download Continuation as MIDI
#@markdown Download performance (primer + generated continuation)
#@markdown as MIDI (optional).

note_seq.sequence_proto_to_midi_file(
    continuation_ns, '/tmp/continuation.mid')
files.download('/tmp/continuation.mid')

"""# Melody-Conditioned Piano Performance Model"""

#@title Setup and Load Checkpoint
#@markdown Set up generation from a melody-conditioned
#@markdown Transformer model.

model_name = 'transformer'
hparams_set = 'transformer_tpu'
ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/melody_conditioned_model_16.ckpt'

class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = MelodyToPianoPerformanceProblem()
melody_conditioned_encoders = problem.get_feature_encoders()

# Set up HParams.
hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
trainer_lib.add_problem_hparams(hparams, problem)
hparams.num_hidden_layers = 16
hparams.sampling_method = 'random'

# Set up decoding HParams.
decode_hparams = decoding.decode_hparams()
decode_hparams.alpha = 0.0
decode_hparams.beam_size = 1

# Create Estimator.
run_config = trainer_lib.create_run_config(hparams)
estimator = trainer_lib.create_estimator(
    model_name, hparams, run_config,
    decode_hparams=decode_hparams)

# These values will be changed by the following cell.
inputs = []
decode_length = 0

# Create input generator.
def input_generator():
  global inputs
  while True:
    yield {
        'inputs': np.array([[inputs]], dtype=np.int32),
        'targets': np.zeros([1, 0], dtype=np.int32),
        'decode_length': np.array(decode_length, dtype=np.int32)
    }

# Start the Estimator, loading from the specified checkpoint.
input_fn = decoding.make_input_fn_from_generator(input_generator())
melody_conditioned_samples = estimator.predict(
    input_fn, checkpoint_path=ckpt_path)

# "Burn" one.
_ = next(melody_conditioned_samples)

#@title Choose Melody
#@markdown Here you can choose a melody to be accompanied by the
#@markdown model.  We have provided a few, or you can upload a
#@markdown MIDI file; if your MIDI file is polyphonic, the notes
#@markdown with highest pitch will be used as the melody.

# Tokens to insert between melody events.
event_padding = 2 * [note_seq.MELODY_NO_EVENT]

melodies = {
    'Mary Had a Little Lamb': [
        64, 62, 60, 62, 64, 64, 64, note_seq.MELODY_NO_EVENT,
        62, 62, 62, note_seq.MELODY_NO_EVENT,
        64, 67, 67, note_seq.MELODY_NO_EVENT,
        64, 62, 60, 62, 64, 64, 64, 64,
        62, 62, 64, 62, 60, note_seq.MELODY_NO_EVENT,
        note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT
    ],
    'Row Row Row Your Boat': [
        60, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        60, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        60, note_seq.MELODY_NO_EVENT, 62,
        64, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        64, note_seq.MELODY_NO_EVENT, 62,
        64, note_seq.MELODY_NO_EVENT, 65,
        67, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        72, 72, 72, 67, 67, 67, 64, 64, 64, 60, 60, 60,
        67, note_seq.MELODY_NO_EVENT, 65,
        64, note_seq.MELODY_NO_EVENT, 62,
        60, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT,
        note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT, note_seq.MELODY_NO_EVENT
    ],
    'Twinkle Twinkle Little Star': [
        60, 60, 67, 67, 69, 69, 67, note_seq.MELODY_NO_EVENT,
        65, 65, 64, 64, 62, 62, 60, note_seq.MELODY_NO_EVENT,
        67, 67, 65, 65, 64, 64, 62, note_seq.MELODY_NO_EVENT,
        67, 67, 65, 65, 64, 64, 62, note_seq.MELODY_NO_EVENT,
        60, 60, 67, 67, 69, 69, 67, note_seq.MELODY_NO_EVENT,
        65, 65, 64, 64, 62, 62, 60, note_seq.MELODY_NO_EVENT
    ]
}

melody = 'Twinkle Twinkle Little Star'  #@param ['Mary Had a Little Lamb', 'Row Row Row Your Boat', 'Twinkle Twinkle Little Star', 'Upload your own!']

if melody == 'Upload your own!':
  # Extract melody from user-uploaded MIDI file.
  melody_ns = upload_midi()
  melody_instrument = note_seq.infer_melody_for_sequence(melody_ns)
  notes = [note for note in melody_ns.notes
           if note.instrument == melody_instrument]
  del melody_ns.notes[:]
  melody_ns.notes.extend(
      sorted(notes, key=lambda note: note.start_time))
  for i in range(len(melody_ns.notes) - 1):
    melody_ns.notes[i].end_time = melody_ns.notes[i + 1].start_time
  inputs = melody_conditioned_encoders['inputs'].encode_note_sequence(
      melody_ns)
else:
  # Use one of the provided melodies.
  events = [event + 12 if event != note_seq.MELODY_NO_EVENT else event
            for e in melodies[melody]
            for event in [e] + event_padding]
  inputs = melody_conditioned_encoders['inputs'].encode(
      ' '.join(str(e) for e in events))
  melody_ns = note_seq.Melody(events).to_sequence(qpm=150)

# Play and plot the melody.
note_seq.play_sequence(
    melody_ns,
    synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(melody_ns)

#@title Generate Accompaniment for Melody
#@markdown Generate a piano performance consisting of the chosen
#@markdown melody plus accompaniment.

# Generate sample events.
decode_length = 4096
sample_ids = next(melody_conditioned_samples)['outputs']

# Decode to NoteSequence.
midi_filename = decode(
    sample_ids,
    encoder=melody_conditioned_encoders['targets'])
accompaniment_ns = note_seq.midi_file_to_note_sequence(midi_filename)

# Play and plot.
note_seq.play_sequence(
    accompaniment_ns,
    synth=note_seq.fluidsynth, sample_rate=SAMPLE_RATE, sf2_path=SF2_PATH)
note_seq.plot_sequence(accompaniment_ns)

#@title Download Accompaniment as MIDI
#@markdown Download accompaniment performance as MIDI (optional).

note_seq.sequence_proto_to_midi_file(
    accompaniment_ns, '/tmp/accompaniment.mid')
files.download('/tmp/accompaniment.mid')