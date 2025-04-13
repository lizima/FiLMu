import os
import json
import librosa
import math

from music21 import converter


def process(root_folder, output_folder):
    genres = {
        "ChaChaCha": "Latin",
        "Jive": "Swing",
        "Quickstep": "Swing",
        "Rumba-American": "Latin",
        "Rumba-International": "Latin",
        "Rumba-Misc": "Latin (miscellaneous)",
        "Samba": "Latin",
        "Tango": "Latin",
        "VienneseWaltz": "Waltz",
        "Waltz": "Waltz"
    }

    ballroom_data = os.path.join(root_folder, "BallroomData")
    ballroom_annotations = os.path.join(root_folder, "BallroomAnnotations", "ballroomGroundTruth")
    res = []
    for dance_style in os.listdir(ballroom_data):
        folder = os.path.join(ballroom_data, dance_style)
        if not os.path.isdir(folder):
            continue
        for song in os.listdir(folder):
            if not str.endswith(song, ".wav"):
                continue
            path = os.path.join(folder, song)
            tempo_path = os.path.join(ballroom_annotations, str.replace(song, ".wav", ".bpm"))
            with open(tempo_path, "r") as f:
                tempo = float(f.read())
                tempo = int(tempo)
            wav, sr = librosa.load(path)

            segments = [{
                # "duration": len(wav) / sr,
                "onset": 0,
                "offset": len(wav) / sr,
                "tempo mean": tempo,
                # "tags": ["ballroom", dance_style],
                "mark": 'M',
            }]
            if dance_style in genres:
               segments[0]["genre"] = genres[dance_style]

            res.append({
                "filename": path,
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
# import math

# from music21 import converter


# def process(root_folder, output_folder):
#     genres = {
#         "ChaChaCha": "Latin",
#         "Jive": "Swing",
#         "Quickstep": "Swing",
#         "Rumba-American": "Latin",
#         "Rumba-International": "Latin",
#         "Rumba-Misc": "Latin (miscellaneous)",
#         "Samba": "Latin",
#         "Tango": "Latin",
#         "VienneseWaltz": "Waltz",
#         "Waltz": "Waltz"
#     }

#     ballroom_data = os.path.join(root_folder, "BallroomData")
#     ballroom_annotations = os.path.join(root_folder, "BallroomAnnotations", "ballroomGroundTruth")
#     res = []
#     for dance_style in os.listdir(ballroom_data):
#         folder = os.path.join(ballroom_data, dance_style)
#         if not os.path.isdir(folder):
#             continue
#         for song in os.listdir(folder):
#             if not str.endswith(song, ".wav"):
#                 continue
#             path = os.path.join(folder, song)
#             tempo_path = os.path.join(ballroom_annotations, str.replace(song, ".wav", ".bpm"))
#             with open(tempo_path, "r") as f:
#                 tempo = float(f.read())
#             wav, sr = librosa.load(path)
#             res.append({
#                 "filename": path,
#                 "samplerate": sr,
#                 "duration": len(wav) / sr,
#                 "tempo mean": tempo,
#                 "tags": ["ballroom", dance_style],
#             })
#             #if dance_style in genres:
#             #    res[-1]["genre"] = genres[dance_style]
#             print(res[-1])
#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#          json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))



