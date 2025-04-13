import os
import json
import librosa
import math

def get_f0(path):
    with open(path, "r") as f:
        lines = f.readlines()
    lines = ["\t".join(line.rstrip().split("\t")) for line in lines]
    return lines

def get_midi(path):
    with open(path, "r") as f:
        lines = f.readlines()
    # delete the empty lines
    lines = [line for line in lines if line != "\n"]
    lines = ["\t".join(line.rstrip().split("\t")) for line in lines]
    new_lines = []
    for line in lines:
        new_line = "\t".join(line.rstrip().split("\t"))
        if len(new_line.split("\t")) != 3:
            new_line = "\t".join(line.rstrip().split(" "))
        new_lines.append(new_line)
    notes = []
    for line in new_lines:
        start, hz, duration = line.rstrip().split("\t")
        midi_note = hz_to_midi(float(hz))
        notes.append({
            "pitch": midi_note,
            "time": start,
            "duration": duration
        })
    return [{
        "source": 1,
        "instrument": "vocal",
        "notes": notes
    }]

def hz_to_midi(hz):
    return 69 + 12 * math.log2(hz / 440)

def process(root_folder, output_folder):
    """
    Process the Choral Singing Dataset
    """

    csd_dataset = os.path.join(root_folder, "ChoralSingingDataset")

    titles = {
        "LI": "Locus Iste",
        "ER": "El Rossinyol",
        "ND": "Niño Dios d'Amor Herido"
    }
    vocal_genders = {
        "soprano": "female",
        "alto": "female",
        "tenor": "male",
        "bass": "male"
    }

    all_files = os.listdir(csd_dataset)
    all_wavs = [f for f in all_files if f.endswith(".wav")]

    res = []
    for wav_path in all_wavs:

        path = os.path.join(csd_dataset, wav_path)

        title_abbr = wav_path.split("_")[1]
        title = titles[title_abbr]

        vocal_part = wav_path.split("_")[2]
        vocal_gender = vocal_genders[vocal_part]

        notes_anno_path = "CSD_" + title_abbr + "_" + vocal_part + "_notes.lab"
        notes_output_path = os.path.join(output_folder, "midis")
        os.makedirs(notes_output_path, exist_ok=True)
        midis = os.path.join(notes_output_path, notes_anno_path + ".json")
        with open(midis, "w") as jsonfile:
            json.dump(get_midi(os.path.join(csd_dataset, notes_anno_path)), jsonfile, indent=2)

        f0_anno_path = wav_path.replace(".wav", ".f0")
        f0_output_path = os.path.join(output_folder, "f0s")
        os.makedirs(f0_output_path, exist_ok=True)
        f0 = os.path.join(f0_output_path, f0_anno_path + ".lst")
        with open(f0, "w") as f:
            f.write("\n".join(get_f0(os.path.join(csd_dataset, f0_anno_path))))

        wav, sr = librosa.load(path)
        res.append({
            "filename": path,
            "samplerate": sr,
            "duration": len(wav) / sr,
            "title": title,
            "genre": "choral",
            "vocal part": vocal_part,
            "monophonic ?": "yes",
            "vocal gender": vocal_gender,
            "f0": f0,
            "midis": midis
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