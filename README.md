# AI MIDI Chords Generator

This is a Generative Model that predicts the next few chords based on a given initial chord

# Google Colab

The code for this Chord Prediction transformer model is available here

-   ([Pre-Trained Model ](https://colab.research.google.com/drive/1ffQt5xyuNoEr9Qf7pkTAkUNTlOrTsRrQ?usp=sharing)) If you want to play around with the already trained model
-   ([Model code](https://colab.research.google.com/drive/1dwB3Bz1uY49B0ljRjU6Gu6cGzWtUMqMQ?usp=sharing)) If you want to see what the code looks like

# `Dev` Installation

1. Clone this repo

```sh
git clone https://github.com/rp-bot/MIDI-gen-ai.git
cd MIDI-gen-ai
```

2. Install using pipenv
    > Note: You need the latest version of Python and pipenv installed.

```sh
pipenv install
```

Everything should work. If not, please create an issue. I'd be happy to help!

# Data Sources

This AI uses a cluster of MIDI files collected from various sources.

> You can take a look at [this repo](https://github.com/rp-bot/Ultimate-MIDI-Scraper) if you want to see how I collected it.

-   https://freemidi.org
-   https://www.midiworld.com/
-   https://www.mididb.com/
