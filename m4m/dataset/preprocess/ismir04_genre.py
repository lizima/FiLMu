import os
import csv
import json
import librosa

# samplerate
# artist
# title
# filename
# genre
# duration
# tags

def process(root_folder, output_folder):
    """
    Process the ISMIR04 Genre dataset.
    """
    ismir04_genre_dataset = os.path.join(root_folder, "ismir04_genre")

    audio_dir = os.path.join(ismir04_genre_dataset, "audio")
    metadata_dir = os.path.join(ismir04_genre_dataset, "metadata")
    splits = ["training", "development", "evaluation"]

    black_list = []
    res = []
    for split in splits:
        metadata_path = os.path.join(metadata_dir, split, "tracklist.csv")
        with open(metadata_path, "r") as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                genre = " ".join(row[0].split("_"))
                artist = " ".join(row[1].split("_"))
                album = " ".join(row[2].split("_"))
                track = " ".join(row[3].split("_"))
                file_path = row[5]
                tags = [genre, artist, album]

                path = os.path.join(audio_dir, split, file_path)
                try:
                    wav, sr = librosa.load(path)
                except:
                    print("Error loading", path)
                    black_list.append(path)
                    continue
                res.append({
                    "filename": path,
                    "samplerate": sr,
                    "duration": len(wav) / sr,
                    "title": track,
                    "genre": genre,
                    "artist": artist,
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

    print(black_list)