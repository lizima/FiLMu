import os
import json
import math
import csv

from m4m.dataset.preprocess.utils import parse_xml, parse_score, get_beats_by_measure


def get_f0(path):
    with open(path, "r") as f:
        lines = f.readlines()[1:]
    lines = ["\t".join(line.rstrip().split(",")) for line in lines]
    return lines


def get_midi(path):
    # print(path)
    with open(path, "r", encoding='latin1') as f:
        lines = f.readlines()[1:]
    lines = [line.rstrip().split(",") for line in lines]
    notes = [{"pitch": round(69 + 12 * math.log2(float(line[1]) / 440)),
              "time": line[0],
              "duration": line[2]} for line in lines]
    return [{
        "source": 1,
        "instrument": "HuQin",
        "notes": notes
    }]


def read_metadata(path, audio_folder):
    metadata = {}
    with open(path, newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)

        for row in csv_reader:
            name = row[0]
            filename = row[1]
            filename = str.replace(filename, "Bang-zi", "Bang_zi")
            filename = str.replace(filename, "Diao_part", "Diao-part")
            region = row[2]
            date = row[3]
            instrument = row[4]
            duration = row[5]
            performer = row[6]
            composers = row[7]
            description = row[8]
            folder = str.replace(instrument, " ", "")

            audio_path = os.path.join(audio_folder, folder, filename, filename + ".wav")

            if not os.path.exists(audio_path):
                audio_path = str.replace(path, "Erhu","Erhu-1")
            if not os.path.exists(audio_path):
                audio_path = str.replace(path, "Erhu","Erhu-2")
            if not os.path.exists(audio_path):
                audio_path = str.replace(path, "Erhu","Erhu-3")

            path = audio_path

            ms = [float(s) for s in duration.split(":")]
            metadata[filename] = {
                "filename": path,
                "duration": ms[0] * 60 + ms[1],
                # "description": description,
                "instruments": [instrument],
                "monophonic ?": "yes",
                # "tags": ["traditional Chinese bowed string instruments", "HuQin"],
                "genre": "Chinese folk",
            }

    return metadata

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
        else:
            segments[0][k] = v
    new_dic['segments'] = segments
    return new_dic

def process(root_folder, output_folder):
    res = []
    metadata_path = os.path.join(root_folder, "metadata-v2.0.csv")
    audio_folder = os.path.join(root_folder, "audios")
    metadata = read_metadata(metadata_path, audio_folder)

    for song in metadata:


        path = str.replace(metadata[song]["filename"], ".wav", ".musicxml")
        score = parse_xml(path)[0][1]
        beats, key_modes, tempos = parse_score(score)
        tempo_key = "tempo" if len(tempos.split(" - ")) > 1 else "tempo mean"

        metadata[song]["time signature"] = beats
        metadata[song]["key mode"] = key_modes
        if not tempos == "":
            metadata[song][tempo_key] = tempos
        metadata[song]["monophonic ?"] = "yes"

        # metadata[song]["beats by measure"] = get_beats_by_measure(score)

        midi_path = os.path.join(output_folder, "midis")
        os.makedirs(midi_path, exist_ok=True)
        # metadata[song]["midi"] = os.path.join(midi_path, str.replace(song, ".musicxml", ".json"))
        # with open(metadata[song]["midi"], "w") as jsonfile:
        #     json.dump(get_midi(str.replace(path, ".musicxml", "-onset.csv")), jsonfile, indent=2)

        f0_path = os.path.join(output_folder, "f0s")
        os.makedirs(f0_path, exist_ok=True)
        # metadata[song]["f0"] = os.path.join(f0_path, str.replace(song, ".musicxml", ".lst"))
        # with open(metadata[song]["f0"], "w") as f:
        #     f.write("\n".join(get_f0(str.replace(path, ".musicxml", "-pitch.csv"))))

        # print(metadata[song])
        metadata_res = rearange(metadata[song])
        print(metadata_res)
        # res.append(metadata[song])
        res.append(metadata_res)

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
# import math
# import csv

# from m4m.dataset.preprocess.utils import parse_xml, parse_score, get_beats_by_measure


# def get_f0(path):
#     with open(path, "r") as f:
#         lines = f.readlines()[1:]
#     lines = ["\t".join(line.rstrip().split(",")) for line in lines]
#     return lines


# def get_midi(path):
#     print(path)
#     with open(path, "r", encoding='latin1') as f:
#         lines = f.readlines()[1:]
#     lines = [line.rstrip().split(",") for line in lines]
#     notes = [{"pitch": round(69 + 12 * math.log2(float(line[1]) / 440)),
#               "time": line[0],
#               "duration": line[2]} for line in lines]
#     return [{
#         "source": 1,
#         "instrument": "HuQin",
#         "notes": notes
#     }]


# def read_metadata(path, audio_folder):
#     metadata = {}
#     with open(path, newline='', encoding='utf-8') as csvfile:
#         csv_reader = csv.reader(csvfile)
#         next(csv_reader)

#         for row in csv_reader:
#             name = row[0]
#             filename = row[1]
#             filename = str.replace(filename, "Bang-zi", "Bang_zi")
#             filename = str.replace(filename, "Diao_part", "Diao-part")
#             region = row[2]
#             date = row[3]
#             instrument = row[4]
#             duration = row[5]
#             performer = row[6]
#             composers = row[7]
#             description = row[8]
#             folder = str.replace(instrument, " ", "")

#             audio_path = os.path.join(audio_folder, folder, filename, filename + ".wav")

#             if not os.path.exists(audio_path):
#                 audio_path = str.replace(path, "Erhu","Erhu-1")
#             if not os.path.exists(audio_path):
#                 audio_path = str.replace(path, "Erhu","Erhu-2")
#             if not os.path.exists(audio_path):
#                 audio_path = str.replace(path, "Erhu","Erhu-3")

#             path = audio_path

#             ms = [float(s) for s in duration.split(":")]
#             metadata[filename] = {
#                 "filename": path,
#                 "duration": ms[0] * 60 + ms[1],
#                 "description": description,
#                 "instruments": [instrument],
#                 "monophonic ?": "yes",
#                 "tags": ["traditional Chinese bowed string instruments", "HuQin"],
#                 "genre": "Chinese folk",
#             }

#     return metadata


# def process(root_folder, output_folder):
#     res = []
#     metadata_path = os.path.join(root_folder, "metadata-v2.0.csv")
#     audio_folder = os.path.join(root_folder, "audios")
#     metadata = read_metadata(metadata_path, audio_folder)

#     for song in metadata:


#         path = str.replace(metadata[song]["filename"], ".wav", ".musicxml")
#         score = parse_xml(path)[0][1]
#         beats, key_modes, tempos = parse_score(score)
#         tempo_key = "tempo" if len(tempos.split(" - ")) > 1 else "tempo mean"

#         metadata[song]["time signature"] = beats
#         metadata[song]["key mode"] = key_modes
#         if not tempos == "":
#             metadata[song][tempo_key] = tempos
#         metadata[song]["monophonic ?"] = "yes"

#         metadata[song]["beats by measure"] = get_beats_by_measure(score)

#         midi_path = os.path.join(output_folder, "midis")
#         os.makedirs(midi_path, exist_ok=True)
#         metadata[song]["midi"] = os.path.join(midi_path, str.replace(song, ".musicxml", ".json"))
#         with open(metadata[song]["midi"], "w") as jsonfile:
#             json.dump(get_midi(str.replace(path, ".musicxml", "-onset.csv")), jsonfile, indent=2)

#         f0_path = os.path.join(output_folder, "f0s")
#         os.makedirs(f0_path, exist_ok=True)
#         metadata[song]["f0"] = os.path.join(f0_path, str.replace(song, ".musicxml", ".lst"))
#         with open(metadata[song]["f0"], "w") as f:
#             f.write("\n".join(get_f0(str.replace(path, ".musicxml", "-pitch.csv"))))

#         print(metadata[song])
#         res.append(metadata[song])

#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#         json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))
