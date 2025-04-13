import os
import json
import librosa
from m4m.dataset.preprocess.utils import parse_xml, parse_score, get_beats_by_measure, is_swing_tempo
import re

def rearange(dic):
    new_dic = {}
    segments = [{'mark': 'M'}]
    for k, v in dic.items():
        # if k == 'filename' or k == 'samplerate':
        if k == 'filename':
            new_dic[k] = v
        elif k == 'duration':
            # new_dic[k] = v
            # segments[0][k] = v
            segments[0]['onset'] = 0
            segments[0]['offset'] = v
        elif k == "title" or k == "samplerate" or k == "beats by measure":
            continue
        else:
            segments[0][k] = v
    new_dic['segments'] = segments
    return new_dic

def format(x):
    return str.replace(x, "_", " ")

def clean_xml(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = re.sub(r'[^\x00-\x7F]', '', content)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(content)

def process(root_folder, output_folder):
    meatadata = os.path.join(root_folder, "meta.csv")
    with open(meatadata, "r") as f:
        lines = f.readlines()[1:]
    lines = [line.rstrip().split(",") for line in lines]
    data = {}
    for line in lines:
        if line[0] not in data:
            data[line[0]] = {}
        path = os.path.join(root_folder, line[8], line[0], line[9] + ".wav")
        wav, sr = librosa.load(path)

        data[line[0]][line[9]] = {
            "filename": path,
            "title": line[0],
            "samplerate": sr,
            "duration": len(wav) / sr,
            "genre": line[8],
            "vocal part": format(line[9]),
            "vocal gender": line[11],
            "monophonic ?": "yes",
        }
        print(data[line[0]][line[9]])

    res = []
    composition = []
    global v, is_swing, key_modes, beats_by_measure, beats, tempos
    for song in data:
        path = data[song]["lead_vocal"]["filename"]
        name = path.split("/")[-2]
        path = os.path.join("/".join(path.split("/")[:-1]), "mixture.wav")
        wav, sr = librosa.load(path)
        xml_path = str.replace(path, "mixture.wav", name + "_SVS.musicxml")

        # ========== new_add ==========
        input_path = xml_path
        output_path = str.replace(xml_path, "SVS", "SVS2")
        clean_xml(input_path, output_path)
        xml_path = output_path
        print(xml_path)
        # ========== new_add ==========
        scores = parse_xml(xml_path)

        for part_name, score in scores:
            beats, key_modes, tempos = parse_score(score)
            beats_by_measure = get_beats_by_measure(score)
            is_swing = is_swing_tempo(xml_path)
            break
        tempo_key = "tempo" if len(tempos.split(" - ")) > 1 else "tempo mean"
        res.append({
            "filename": path,
            # "title": data[song]["lead_vocal"]["title"],
            # "samplerate": sr,
            "duration": len(wav) / sr,
            "genre": data[song]["lead_vocal"]["genre"],
            "monophonic ?": "no",
            "swing ?": is_swing,
            # "tags": ["Acappella"],
            "key mode": key_modes,
            "time signature": beats,
            # "beats by measure": beats_by_measure,

        })
        if not tempos == "":
            res[-1][tempo_key] = tempos

        res[-1] = rearange(res[-1])
        print(res[-1])
        coms = {}

        for v in data[song]:
            d = data[song][v]
            if not v == "vocal_percussion":
                d["swing ?"] = is_swing
                d["key mode"] = key_modes
                d["time signature"] = beats
                d["beats by measure"] = beats_by_measure
            res.append(d)
            if not tempos == "":
                res[-1][tempo_key] = tempos
            res[-1] = rearange(res[-1])
            print(res[-1])
            coms[data[song][v]["vocal part"]] = data[song][v]["filename"]
        composition.append(
            {"mix": path,
             "samplerate": sr,
            "duration": len(wav) / sr,
             "tracks": coms}
        )

    with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
        json.dump(res, jsonfile, indent=2)
    with open(os.path.join(output_folder, "composition.json"), 'w') as jsonfile:
        json.dump(composition, jsonfile, indent=2)

    keys = {}
    for data in res:
        for key in data:
            keys[key] = 0
    with open(os.path.join(output_folder, "keys.lst"), "w") as f:
        f.write("\n".join([k for k in keys]))


# import os
# import json
# import librosa
# from m4m.dataset.preprocess.utils import parse_xml, parse_score, get_beats_by_measure, is_swing_tempo
# import re


# def format(x):
#     return str.replace(x, "_", " ")

# def clean_xml(file_path, output_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         content = file.read()

#     content = re.sub(r'[^\x00-\x7F]', '', content)

#     with open(output_path, 'w', encoding='utf-8') as file:
#         file.write(content)

# def process(root_folder, output_folder):
#     meatadata = os.path.join(root_folder, "meta.csv")
#     with open(meatadata, "r") as f:
#         lines = f.readlines()[1:]
#     lines = [line.rstrip().split(",") for line in lines]
#     data = {}
#     for line in lines:
#         if line[0] not in data:
#             data[line[0]] = {}
#         path = os.path.join(root_folder, line[8], line[0], line[9] + ".wav")
#         wav, sr = librosa.load(path)

#         data[line[0]][line[9]] = {
#             "filename": path,
#             "title": line[0],
#             "samplerate": sr,
#             "duration": len(wav) / sr,
#             "genre": line[8],
#             "vocal part": format(line[9]),
#             "vocal gender": line[11],
#             "monophonic ?": "yes",
#         }
#         print(data[line[0]][line[9]])

#     res = []
#     composition = []
#     global v, is_swing, key_modes, beats_by_measure, beats, tempos
#     for song in data:
#         path = data[song]["lead_vocal"]["filename"]
#         name = path.split("/")[-2]
#         path = os.path.join("/".join(path.split("/")[:-1]), "mixture.wav")
#         wav, sr = librosa.load(path)
#         xml_path = str.replace(path, "mixture.wav", name + "_SVS.musicxml")

#         # ========== new_add ==========
#         input_path = xml_path
#         output_path = str.replace(xml_path, "SVS", "SVS2")
#         clean_xml(input_path, output_path)
#         xml_path = output_path
#         # ========== new_add ==========

#         scores = parse_xml(xml_path)
#         for part_name, score in scores:
#             beats, key_modes, tempos = parse_score(score)
#             beats_by_measure = get_beats_by_measure(score)
#             is_swing = is_swing_tempo(xml_path)
#             break
#         tempo_key = "tempo" if len(tempos.split(" - ")) > 1 else "tempo mean"
#         res.append({
#             "filename": path,
#             "title": data[song]["lead_vocal"]["title"],
#             "samplerate": sr,
#             "duration": len(wav) / sr,
#             "genre": data[song]["lead_vocal"]["genre"],
#             "monophonic ?": "no",
#             "swing ?": is_swing,
#             "tags": ["Acappella"],
#             "key mode": key_modes,
#             "time signature": beats,
#             "beats by measure": beats_by_measure,

#         })
#         if not tempos == "":
#             res[-1][tempo_key] = tempos

#         #
#         print(res[-1])
#         coms = {}

#         for v in data[song]:
#             d = data[song][v]
#             if not v == "vocal_percussion":
#                 d["swing ?"] = is_swing
#                 d["key mode"] = key_modes
#                 d["time signature"] = beats
#                 d["beats by measure"] = beats_by_measure
#             res.append(d)
#             if not tempos == "":
#                 res[-1][tempo_key] = tempos
#             print(res[-1])
#             coms[data[song][v]["vocal part"]] = data[song][v]["filename"]
#         composition.append(
#             {"mix": path,
#              "samplerate": sr,
#             "duration": len(wav) / sr,
#              "tracks": coms}
#         )

#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#         json.dump(res, jsonfile, indent=2)
#     with open(os.path.join(output_folder, "composition.json"), 'w') as jsonfile:
#         json.dump(composition, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))

