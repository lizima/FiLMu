import os
import json
import librosa


def process(root_folder, output_folder):
    res = []
    data = {}
    for song in os.listdir(root_folder):
        if not str.endswith(song, ".csv"):
            continue
        song = song.split(".csv")[0].split("_tech_")
        if len(song) == 2:
            name = song[0]
            tag = song[-1]
        else:
            tag = song[0].split("_")[-1]
            name = song[0]

        tag = "Flutter-tongue" if tag == "FT" else tag
        if name in data:
            data[name].append(tag)
        else:
            data[name] = [tag]

    for name in data:
        filename = os.path.join(root_folder, name + ".wav")
        wav, sr = librosa.load(filename)
        segments = [{
            # "duration": len(wav) / sr,
            "onset": 0,
            "offset": len(wav) / sr,
            # "tags": data[name],
            "instruments": "Chinese Bamboo Flute",
            "monophonic ?": 'yes',
            "mark": 'M',
        }]

        res.append({
            "filename": filename,
            # "samplerate": sr,
            # "duration": len(wav) / sr,
            "segments": segments
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


# import os
# import json
# import librosa


# def process(root_folder, output_folder):
#     res = []
#     data = {}
#     for song in os.listdir(root_folder):
#         if not str.endswith(song, ".csv"):
#             continue
#         song = song.split(".csv")[0].split("_tech_")
#         if len(song) == 2:
#             name = song[0]
#             tag = song[-1]
#         else:
#             tag = song[0].split("_")[-1]
#             name = song[0]

#         tag = "Flutter-tongue" if tag == "FT" else tag
#         if name in data:
#             data[name].append(tag)
#         else:
#             data[name] = [tag]

#     for name in data:
#         filename = os.path.join(root_folder, name + ".wav")
#         wav, sr = librosa.load(filename)
#         res.append({
#             "filename": filename,
#             "samplerate": sr,
#             "duration": len(wav) / sr,
#             "tags": data[name],
#             "instruments": "Chinese Bamboo Flute",
#             "monophonic ?": "yes"
#         })
#         print(res[-1])

#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#          json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))
