import os
import json
import librosa

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
        elif k == 'artist' or k == 'title' or k == 'beats by measure':
            continue
        else:
            segments[0][k] = v
    new_dic['segments'] = segments
    return new_dic

def process(root_folder, output_folder):
    meta_data_path = os.path.join(root_folder, "stats.csv")
    res = []
    with open(meta_data_path, "r") as f:
        rows = f.readlines()
        for i, row in enumerate(rows):
            try:
                row = row.rstrip().split(",")
                if i == 0:
                    heads = row
                else:
                    data = {}
                    idx = 0
                    idx_end = idx + 1

                    for h in heads:
                        val = str.replace(row[idx], '""', "-")
                        if '"' in val:
                            val = str.replace(row[idx_end], '""', "-")
                            while '"' not in val:
                                idx_end += 1
                                val = str.replace(row[idx_end], '""', "-")
                            idx_end += 1
                        val = ",".join(row[idx: idx_end])
                        if val == "":
                            continue
                        if h == "filename":
                            name = val.split(".wav")[0]
                            val = os.path.join(root_folder, "music", name + ".au")

                            wav, sr = librosa.load(val)
                            # data["samplerate"] = sr
                            data["duration"] = len(wav) / sr
                            data["genre"] = name.split(".")[0]

                        if h == "meter":
                            h = "time signature"
                        if h == "beat by measure":
                            h = "beats by measure"

                        data[h] = val
                        idx = idx_end
                        idx_end = idx + 1
                    # print(data)
                    if "swing confidence" in data and float(data["swing confidence"]) < 0.7:
                        excludes = ["swing confidence", "swing ratio median", "swing ratio iqr", "swing ?"]
                    else:
                        excludes = ["swing confidence"]
                    n_data = {}
                    for k in data:
                        if k in excludes:
                            continue
                        n_data[k] = data[k]

                    print(n_data)
                    n_data = rearange(n_data)
                    res.append(n_data)
            except:
                print(row)
    with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
        json.dump(res, jsonfile, indent=2)

    keys = {"samplerate": 0, "duration": 0}
    for data in res:
        for key in data:
            keys[key] = 0
    with open(os.path.join(output_folder, "keys.lst"), "w") as f:
        f.write("\n".join([k for k in keys]))


# import os
# import json
# import librosa


# def process(root_folder, output_folder):
#     meta_data_path = os.path.join(root_folder, "stats.csv")
#     res = []
#     with open(meta_data_path, "r") as f:
#         rows = f.readlines()
#         for i, row in enumerate(rows):
#             row = row.rstrip().split(",")
#             if i == 0:
#                 heads = row
#             else:
#                 data = {}
#                 idx = 0
#                 idx_end = idx + 1

#                 for h in heads:
#                     val = str.replace(row[idx], '""', "-")
#                     if '"' in val:
#                         val = str.replace(row[idx_end], '""', "-")
#                         while '"' not in val:
#                             idx_end += 1
#                             val = str.replace(row[idx_end], '""', "-")
#                         idx_end += 1
#                     val = ",".join(row[idx: idx_end])
#                     if val == "":
#                         continue
#                     if h == "filename":
#                         name = val.split(".wav")[0]
#                         val = os.path.join(root_folder, "music", name + ".au")

#                         wav, sr = librosa.load(val)
#                         data["samplerate"] = sr
#                         data["duration"] = len(wav) / sr
#                         data["genre"] = name.split(".")[0]

#                     if h == "meter":
#                         h = "time signature"
#                     if h == "beat by measure":
#                         h = "beats by measure"

#                     data[h] = val
#                     idx = idx_end
#                     idx_end = idx + 1
#                 print(data)
#                 if "swing confidence" in data and float(data["swing confidence"]) < 0.7:
#                     excludes = ["swing confidence", "swing ratio median", "swing ratio iqr", "swing ?"]
#                 else:
#                     excludes = ["swing confidence"]
#                 n_data = {}
#                 for k in data:
#                     if k in excludes:
#                         continue
#                     n_data[k] = data[k]
#                 res.append(n_data)
#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#         json.dump(res, jsonfile, indent=2)

#     keys = {"samplerate": 0, "duration": 0}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))



