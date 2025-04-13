import os
import json
import librosa
import pretty_midi as pm

# for tags
tag_data = {
    1: {
        "V": {"genre": "Classical", "instrument": "Solo Piano"},
        "A": {"genre": "Classical", "instrument": "Woodwind quint"},
        "T": {"genre": "Classical", "instrument": "Solo Piano"}
    },
    2: {
        "V": {"genre": "Disco", "instrument": "Disco"},
        "A": {"genre": "Rock/Pop", "instrument": "Rock band"},
        "T": {"genre": "Classical", "instrument": "Brass Quint."}
    },
    3: {
        "V": {"genre": "Swing", "instrument": "Jazz Combo"},
        "A": {"genre": "Rock/Pop", "instrument": "Rock Band"},
        "T": {"genre": "Surf Rock", "instrument": "Rock band"}
    },
    4: {
        "V": {"genre": "Rock/Pop", "instrument": "Rock band"},
        "A": {"genre": "Rock/Hard Rock", "instrument": "Rock band"},
        "T": {"genre": "Rock", "instrument": "Rock Band 60s"}
    },
    5: {
        "V": {"genre": "Piano Rock/Funk", "instrument": "Piano rock"},
        "A": {"genre": "Piano ballad", "instrument": "Rock band"},
        "T": {"genre": "Bluegrass", "instrument": "Bluegrass"}
    },
    6: {
        "V": {"genre": "Soft rock", "instrument": "Rock band"},
        "A": {"genre": "Rock/Hard Rock", "instrument": "Rock band"},
        "T": {"genre": "EuroPop", "instrument": "Synths/Euro"}
    },
    7: {
        "V": {"genre": "60s Rock", "instrument": "Rock band"},
        "A": {"genre": "Swing/Bebop", "instrument": "Jazz Combo"},
        "T": {"genre": "Rock/Pop", "instrument": "Rock Band"}
    },
    8: {
        "V": {"genre": "Latin", "instrument": "Jazz Combo"},
        "A": {"genre": "Swing/Bebop", "instrument": "Jazz Combo"},
        "T": {"genre": "Stadium Rock", "instrument": "Rock Band"}
    },
    9: {
        "V": {"genre": "Ragtime Piano", "instrument": "Solo Piano"},
        "A": {"genre": "Rock", "instrument": "Rock Piano"},
        "T": {"genre": "Folk Rock", "instrument": "Rock Band"}
    },
    10: {
        "V": {"genre": "Classical/Video Game", "instrument": "Orchestra"},
        "A": {"genre": "Classical", "instrument": "Brass quint."},
        "T": {"genre": "Choral/Classical", "instrument": "SATB+Piano"}
    }
}

def get_midi(path):
    midi = pm.PrettyMIDI(path)
    midi_json = []
    for i, instrument in enumerate(midi.instruments):
        inst_name = pm.program_to_instrument_name(instrument.program)
        instrument_json = {
            "source": i + 1,
            "instrument": inst_name
        }
        notes = []
        for note in instrument.notes:
            notes.append({
                "pitch": note.pitch,
                "time": note.start,
                "duration": note.end - note.start
            })
        instrument_json["notes"] = notes
        midi_json.append(instrument_json)
    return midi_json

def process(root_folder, output_folder):
    """
    Process the IsoVAT dataset.

    Genre, instrument, emotion are all contained in the tags.
    """

    audio_dir = os.path.join(root_folder, "Audio")
    midi_dir = os.path.join(root_folder, "MIDI")

    emotion_dims = ["Valence", "Arousal", "Tension"]
    emotion_levels = {"L": "Low", "M": "Medium", "H": "High"}

    # retrieve all audio files
    res = []
    for root, _, files in os.walk(audio_dir):
        for file in files:
            if file.endswith(".wav"):

                path = os.path.join(root, file)

                emotion_type = root.split("\\")[-1]
                emo_level = file.split("-")[2][0] # "L"/"M"/"H"
                emo_tag = emotion_levels[emo_level] + " " + emotion_type

                song_id = file.split("-")[1]
                genre = tag_data[int(song_id)][emotion_type[0]]["genre"]
                instrument = tag_data[int(song_id)][emotion_type[0]]["instrument"]

                tags = []
                tags.append(genre)
                tags.append(instrument)
                tags.append(emo_tag)

                midi_path = os.path.join(midi_dir, emotion_type, file.replace(".wav", ".mid"))
                midis = get_midi(midi_path)

                midis_output_path = os.path.join(output_folder, "midis")
                os.makedirs(midis_output_path, exist_ok=True)
                midis = os.path.join(midis_output_path, file + ".json")
                with open(midis, "w") as jsonfile:
                    json.dump(get_midi(midi_path), jsonfile, indent=2)

                wav, sr = librosa.load(path)
                res.append({
                    "filename": path,
                    "samplerate": sr,
                    "duration": len(wav) / sr,
                    "genre": genre,
                    "emotion": emo_tag,
                    "instrument": instrument,
                    "monophonic ?": "no",
                    "midis": midis,
                    "tags": tags,
                })

                print(res[-1])

    with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
        json.dump(res, jsonfile, indent=2)

    keys = {}
    for data in res:
        for key in data:
            keys[key] = 0
    with open(os.path.join(output_folder, "keys.lst"), "w") as f:
        f.write("\n".join([k for k in keys]))