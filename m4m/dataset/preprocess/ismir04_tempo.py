import os
import json
import librosa

def process(root_folder, output_folder):
    """
    Process the ISMIR 2004 tempo dataset
    """

    ismir04_tempo_dataset = os.path.join(root_folder, "Songs_Data_annotations_Results")

    metadata_file = os.path.join(ismir04_tempo_dataset, "labFiles")
    with open(metadata_file, "r") as f:
        metadata = f.readlines()

    res = []
    for line in metadata:
        line = line.rstrip()
        if not line:
            continue

        line = line[2:]
        anno_file = line.replace(".lab", ".bpm")
        anno_path = os.path.join(ismir04_tempo_dataset, anno_file)

        wav_name = anno_file.replace("Fabien's annotations/", "")[:-9] + ".wav"
        wav_path = os.path.join(ismir04_tempo_dataset, wav_name)

        title = line.split("/")[-1][3:]
        title = title.replace("_", " ")[:-9]

        if not os.path.exists(anno_path) or not os.path.exists(wav_path):
            continue

        with open(anno_path, "r") as f:
            tempo = f.readlines()[0].rstrip()
            tempo = float(tempo)
        wav, sr = librosa.load(wav_path)
        res.append({
            "filename": wav_path,
            "samplerate": sr,
            "duration": len(wav) / sr,
            "title": title,
            "tempo": tempo
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