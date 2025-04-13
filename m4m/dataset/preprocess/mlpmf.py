import os
import json
import librosa

def rearange(dic):
    new_dic = {}
    segments = [{'mark': 'M'}]
    for k, v in dic.items():
        if k == 'filename':
            new_dic[k] = v
        elif k == 'duration':
            # new_dic[k] = v
            # segments[0][k] = v
            segments[0]['onset'] = 0
            segments[0]['offset'] = v
        elif k == "title" or k == "artist" or k == "beats by measure":
            continue
        else:
            segments[0][k] = v
    new_dic['segments'] = segments
    return new_dic

def process(root_folder, output_folder):
    audio_folder = os.path.join(root_folder, "audio")
    metadata = os.path.join(root_folder, "metadata.csv")
    annotations = os.path.join(root_folder, "annotations.csv")
    data = {}
    with open(metadata, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.split(";")
            path = os.path.join(audio_folder, line[0] + ".mp3")
            wav, sr = librosa.load(path)
            data[line[0]] = {
                "filename": path,
                # "samplerate": sr,
                "duration": len(wav) / sr
            }
            if not line[4] == "":
                data[line[0]]["title"] = line[3]
            if not line[5] == "":
                data[line[0]]["artist"] = line[4]
            print(data[line[0]])


    features = ["melodiousness",
                "articulation",
                "rhythmic stability",
                "rhythmic complexity",
                "dissonance",
                "tonal stability",
                "modality"]
    with open(annotations, "r") as f:
        lines = f.readlines()
        for line in lines[1:]:
            line = line.rstrip().split(",")
            for i in range(len(features)):
                data[line[0]][features[i]] = line[i + 1]


    res = [rearange(data[d]) for d in data]


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
#     audio_folder = os.path.join(root_folder, "audio")
#     metadata = os.path.join(root_folder, "metadata.csv")
#     annotations = os.path.join(root_folder, "annotations.csv")
#     data = {}
#     with open(metadata, "r") as f:
#         lines = f.readlines()
#         for line in lines[1:]:
#             line = line.split(";")
#             path = os.path.join(audio_folder, line[0] + ".mp3")
#             wav, sr = librosa.load(path)
#             data[line[0]] = {
#                 "filename": path,
#                 "samplerate": sr,
#                 "duration": len(wav) / sr
#             }
#             if not line[4] == "":
#                 data[line[0]]["title"] = line[3]
#             if not line[5] == "":
#                 data[line[0]]["artist"] = line[4]
#             print(data[line[0]])


#     features = ["melodiousness",
#                 "articulation",
#                 "rhythmic stability",
#                 "rhythmic complexity",
#                 "dissonance",
#                 "tonal stability",
#                 "modality"]
#     with open(annotations, "r") as f:
#         lines = f.readlines()
#         for line in lines[1:]:
#             line = line.rstrip().split(",")
#             for i in range(len(features)):
#                 data[line[0]][features[i]] = line[i + 1]


#     res = [data[d] for d in data]


#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#         json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))



