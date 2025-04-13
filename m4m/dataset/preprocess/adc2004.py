import os
import json
import librosa

def save_pitch(path, out_path):
    with open(path, "r") as f:
        pitch = f.readline()
    pitch = ["\t".join(p.rstrip().split("     ")) for p in pitch]
    with open(out_path, "w") as f:
        f.write("\n".join(pitch))

def process(root_folder, output_folder):
    res = []
    genres = ["jazz", "pop", "opera"]
    for song in os.listdir(root_folder):
        if not str.endswith(song, ".wav"):
            continue
        path = os.path.join(root_folder, song)
        pitch_path = os.path.join(output_folder, "predominant_pitch")
        os.makedirs(pitch_path, exist_ok=True)
        pitch_path = os.path.join(pitch_path, str.replace(song, ".wav", ".txt"))
        in_pitch_path = str.replace(path, "MIDI.wav", "REF.txt")
        in_pitch_path = str.replace(in_pitch_path, ".wav", "REF.txt")
        save_pitch(in_pitch_path, pitch_path)
        wav, sr = librosa.load(path)

        # res.append({
        #             "filename": path,
        #             "samplerate": sr,
        #             "duration": len(wav) / sr,
        #             "predominant pitch": pitch_path
        # })



        segments = [{
            # "duration": len(wav) / sr,
            "onset": 0,
            "offset": len(wav) / sr,
            # "predominant pitch": pitch_path,
            "mark": 'M',
        }]

        for genre in genres:
            if len(song.split(genre)) == 2:
                segments[0]["genre"] = genre

                res.append({ # will not append if genre information does not exist (because no other attributes)
                    "filename": path,
                    # "samplerate": sr,
                    # "duration": len(wav) / sr,
                    "segments": segments
                })
                print(res[-1])

        # for genre in genres:
        #     if len(song.split(genre)) == 2:
        #         res[-1]["genre"] = genre

        

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

# def save_pitch(path, out_path):
#     with open(path, "r") as f:
#         pitch = f.readline()
#     pitch = ["\t".join(p.rstrip().split("     ")) for p in pitch]
#     with open(out_path, "w") as f:
#         f.write("\n".join(pitch))

# def process(root_folder, output_folder):
#     res = []
#     genres = ["jazz", "pop", "opera"]
#     for song in os.listdir(root_folder):
#         if not str.endswith(song, ".wav"):
#             continue
#         path = os.path.join(root_folder, song)
#         pitch_path = os.path.join(output_folder, "predominant_pitch")
#         os.makedirs(pitch_path, exist_ok=True)
#         pitch_path = os.path.join(pitch_path, str.replace(song, ".wav", ".txt"))
#         in_pitch_path = str.replace(path, "MIDI.wav", "REF.txt")
#         in_pitch_path = str.replace(in_pitch_path, ".wav", "REF.txt")
#         save_pitch(in_pitch_path, pitch_path)
#         wav, sr = librosa.load(path)
#         res.append({
#                     "filename": path,
#                     "samplerate": sr,
#                     "duration": len(wav) / sr,
#                     "predominant pitch": pitch_path
#         })

#         for genre in genres:
#             if len(song.split(genre)) == 2:
#                 res[-1]["genre"] = genre

#         print(res[-1])

#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#          json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))


