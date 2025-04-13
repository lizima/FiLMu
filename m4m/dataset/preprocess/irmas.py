import os
import json
import librosa
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
        else:
            segments[0][k] = v
    new_dic['segments'] = segments
    return new_dic

def process(root_folder, output_folder):
    instr_dict = {
        "cel": "cello",
        "cla": "clarinet",
        "flu": "flute",
        "gac": "acoustic guitar",
        "gel": "electric guitar",
        "org": "organ",
        "pia": "piano",
        "sax": "saxophone",
        "tru": "trumpet",
        "vio": "violin",
        "voi": "human singing voice"
    }
    genres = ["Jazz", "Rock", "Blues"]

    genre_dict = {
        'cou_fol': 'country-folk',
        'cla': 'classical',
        'pop_roc': 'pop-rock',
        'lat-soul': 'latin-soul',
    }

    drum_dict = {
        'dru': 'yes',
        'nod': 'no',
    }

    res = []
    for folder in os.listdir(root_folder):
        part_folder = os.path.join(root_folder, folder)
        if not os.path.isdir(part_folder):
            continue
        for sub_folder in os.listdir(part_folder):
            sub_folder = os.path.join(part_folder, sub_folder)
            if not os.path.isdir(sub_folder):
                continue
            for fname in os.listdir(sub_folder):
                path = os.path.join(sub_folder, fname)
                if 'Testing' in path:
                    if not str.endswith(fname, ".txt"):
                        continue
                    with open(path, "r") as f:
                        instrs = f.readlines()
                    instrs = [instr_dict[instr.rstrip()] for instr in instrs]
                    audio_path = str.replace(path, ".txt", ".wav")
                    wav, sr = librosa.load(audio_path)
                    if len(wav) / sr < 8.:
                        continue
                    res.append({
                        "filename": audio_path,
                        # "samplerate": sr,
                        "duration": len(wav) / sr,
                        # "predominant instruments": instrs
                        "instruments": instrs
                    })
                    for genre in genres:
                        if genre in fname.split(" "):
                            res[-1]["genre"] = genre
                            break
                    res[-1] = rearange(res[-1])
                    print(res[-1])
                else:
                    tags = re.findall(r'\[([^\]]+)\]', fname)
                    instrs = []
                    genres = None
                    drums = None
                    for tag in tags:
                        if tag in instr_dict.keys():
                            instrs.append(instr_dict[tag])
                        if tag in genre_dict.keys():
                            genres = genre_dict[tag]
                        if tag in drum_dict.keys():
                            drums = drum_dict[tag]
                    audio_path = path
                    wav, sr = librosa.load(audio_path)
                    if len(wav) / sr < 8.0:
                        # all of the training data are only 3 seconds
                        continue
                    res.append({
                        "filename": audio_path,
                        # "samplerate": sr,
                        "duration": len(wav) / sr,
                        # "predominant instruments": instrs,
                        "instruments": instrs
                    })
                    if genres is not None:
                        res[-1]["genre"] = genres
                    if drums is not None:
                        res[-1]["with_drums"] = drums
                    res[-1] = rearange(res[-1])
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
#     instr_dict = {
#         "cel": "cello",
#         "cla": "clarinet",
#         "flu": "flute",
#         "gac": "acoustic guitar",
#         "gel": "electric guitar",
#         "org": "organ",
#         "pia": "piano",
#         "sax": "saxophone",
#         "tru": "trumpet",
#         "vio": "violin",
#         "voi": "human singing voice"
#     }
#     genres = ["Jazz", "Rock", "Blues"]
#     res = []
#     for folder in os.listdir(root_folder):
#         part_folder = os.path.join(root_folder, folder)
#         if not os.path.isdir(part_folder):
#             continue
#         for sub_folder in os.listdir(part_folder):
#             sub_folder = os.path.join(part_folder, sub_folder)
#             if not os.path.isdir(sub_folder):
#                 continue
#             for fname in os.listdir(sub_folder):
#                 path = os.path.join(sub_folder, fname)
#                 if not str.endswith(fname, ".txt"):
#                     continue
#                 with open(path, "r") as f:
#                     instrs = f.readlines()
#                 instrs = [instr_dict[instr.rstrip()] for instr in instrs]
#                 audio_path = str.replace(path, ".txt", ".wav")
#                 wav, sr = librosa.load(audio_path)
#                 if len(wav) / sr < 8.:
#                     continue
#                 res.append({
#                     "filename": audio_path,
#                     "samplerate": sr,
#                     "duration": len(wav) / sr,
#                     "predominant instruments": instrs
#                 })
#                 for genre in genres:
#                     if genre in fname.split(" "):
#                         res[-1]["genre"] = genre
#                         break
#                 print(res[-1])

#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#          json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))


