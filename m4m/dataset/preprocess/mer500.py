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
            segments[0]['onset'] = 0
            segments[0]['offset'] = v
        else:
            segments[0][k] = v
    new_dic['segments'] = segments
    return new_dic

def process(root_folder, output_folder):
    res = []
    for emotion in os.listdir(root_folder):
        folder = os.path.join(root_folder, emotion, emotion)
        for song in os.listdir(folder):
            path = os.path.join(folder, song)
            wav, sr = librosa.load(path)
            res.append(
                {
                    "filename": path,
                    # "samplerate": sr,
                    "duration": len(wav) / sr,
                    "emotion": emotion
                })
            
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
#     res = []
#     for emotion in os.listdir(root_folder):
#         folder = os.path.join(root_folder, emotion, emotion)
#         for song in os.listdir(folder):
#             path = os.path.join(folder, song)
#             wav, sr = librosa.load(path)
#             res.append(
#                 {
#                     "filename": path,
#                     "samplerate": sr,
#                     "duration": len(wav) / sr,
#                     "emotion": emotion
#                 })
#             print(res[-1])

#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#          json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))


