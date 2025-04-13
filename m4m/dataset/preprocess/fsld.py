import os
import json
import librosa

# def rearange(dic):
#     new_dic = {}
#     segments = [{'mark': 'M'}]
#     for k, v in dic.items():
#         if k == 'filename' or k == 'samplerate':
#             new_dic[k] = v
#         elif k == 'duration':
#             new_dic[k] = v
#             segments[0][k] = v
#         else:
#             segments[0][k] = v
#     new_dic['segments'] = segments
#     return new_dic

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
    ac_analysis_folder = os.path.join(root_folder, "ac_analysis")
    fs_analysis_folder = os.path.join(root_folder, "fs_analysis")

    annotations_folder = os.path.join(root_folder, "annotations")
    audio_folder = os.path.join(root_folder, "audio", "wav")
    data = {}
    for song in os.listdir(ac_analysis_folder):
        print('hi1')
        path = os.path.join(ac_analysis_folder, song)
        name = song.split(".json")[0].split("_analysis")[0] + ".wav.wav"
        filename = os.path.join(audio_folder, name)
        if not os.path.exists(filename):
            name = str.replace(name, ".wav.wav", ".aiff.wav")
            filename = os.path.join(audio_folder, name)
        tid = name.split("_")[0]
        d = {}
        wav, sr = librosa.load(filename)
        dur = len(wav) / sr
        with open(path, "r") as f:
            prop = json.load(f)
            if dur < 10:
                continue
            d["filename"] = filename
            d["duration"] = dur
            # d["samplerate"] = sr
            d["is_loop"] = prop["loop"]
            if prop["tempo_confidence"] > 0.7:
                # d["tempo mean"] = prop["tempo"]
                d["tempo mean"] = int(prop["tempo"])
            if prop["tonality_confidence"] > 0.7:
                # d["key mode"] = str.replace(prop["tonality"], " ", ":")
                d["key mode"] = str.replace(prop["tonality"], " ", "")

        if len(d) > 0:
            data[tid] = d

    for song in os.listdir(fs_analysis_folder):
        print('hi2')
        path = os.path.join(fs_analysis_folder, song)
        tid = song.split(".json")[0]
        if tid not in data:
            continue
        with open(path, "r") as f:
            prop = json.load(f)
            # data[tid]["tags"] = prop["tags"]


    for folder in os.listdir(annotations_folder):
        print('hi3')
        if folder in [".DS_Store"]:
            continue
        folder = os.path.join(annotations_folder, folder)

        for song in os.listdir(folder):

            path = os.path.join(folder, song)
            tid = song.split("sound-")[-1].split(".json")[0]
            if tid not in data:
                continue
            with open(path, "r") as f:
                prop = json.load(f)
                # data[tid]["instrumentation"] = prop["instrumentation"]

                instrumentation_list = []
                for k, v in prop["instrumentation"].items():
                    if v:
                        instrumentation_list.append(k)
                data[tid]["instrumentation"] = instrumentation_list

                if len(prop["genres"]) > 0:
                    data[tid]["genres"] = prop["genres"] if len(prop["genres"]) > 1 else prop["genres"][0]
                if len(prop["genres"]) == 1:
                    data[tid]["genre"] = prop["genres"][0]
                if prop["bpm"] != "" and prop["bpm"] != "none":
                    # data[tid]["tempo mean"] = float(prop["bpm"])
                    data[tid]["tempo mean"] = int(float(prop["bpm"]))
            
            # data[tid] = rearange(data[tid])
            print(data[tid])

    # res = [data[d] for d in data]
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
#     ac_analysis_folder = os.path.join(root_folder, "ac_analysis")
#     fs_analysis_folder = os.path.join(root_folder, "fs_analysis")

#     annotations_folder = os.path.join(root_folder, "annotations")
#     audio_folder = os.path.join(root_folder, "audio", "wav")
#     data = {}
#     for song in os.listdir(ac_analysis_folder):
#         path = os.path.join(ac_analysis_folder, song)
#         name = song.split(".json")[0].split("_analysis")[0] + ".wav.wav"
#         filename = os.path.join(audio_folder, name)
#         if not os.path.exists(filename):
#             name = str.replace(name, ".wav.wav", ".aiff.wav")
#             filename = os.path.join(audio_folder, name)
#         tid = name.split("_")[0]
#         d = {}
#         wav, sr = librosa.load(filename)
#         dur = len(wav) / sr
#         with open(path, "r") as f:
#             prop = json.load(f)
#             if dur < 10:
#                 continue
#             d["filename"] = filename
#             d["duration"] = dur
#             d["samplerate"] = sr
#             d["loop"] = prop["loop"]
#             if prop["tempo_confidence"] > 0.7:
#                 d["tempo mean"] = prop["tempo"]
#             if prop["tonality_confidence"] > 0.7:
#                 d["key mode"] = str.replace(prop["tonality"], " ", ":")

#         if len(d) > 0:
#             data[tid] = d

#     for song in os.listdir(fs_analysis_folder):
#         path = os.path.join(fs_analysis_folder, song)
#         tid = song.split(".json")[0]
#         if tid not in data:
#             continue
#         with open(path, "r") as f:
#             prop = json.load(f)
#             data[tid]["tags"] = prop["tags"]


#     for folder in os.listdir(annotations_folder):
#         if folder in [".DS_Store"]:
#             continue
#         folder = os.path.join(annotations_folder, folder)

#         for song in os.listdir(folder):

#             path = os.path.join(folder, song)
#             tid = song.split("sound-")[-1].split(".json")[0]
#             if tid not in data:
#                 continue
#             with open(path, "r") as f:
#                 prop = json.load(f)
#                 data[tid]["instrumentation"] = prop["instrumentation"]
#                 if len(prop["genres"]) > 0:
#                     data[tid]["genre"] = prop["genres"] if len(prop["genres"]) > 1 else prop["genres"][0]
#                 data[tid]["tempo mean"] = prop["bpm"]
#             print(data[tid])

#     res = [data[d] for d in data]
#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#          json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))



