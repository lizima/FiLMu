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
    root_folder_ = root_folder + '/giantsteps-tempo-dataset/audio'
    res = []
    for song in os.listdir(root_folder_):
        path = os.path.join(root_folder_, song)
        wav, sr = librosa.load(path)

        tempo_anno_dir = root_folder + '/giantsteps-tempo-dataset/annotations/tempo'

        doc_name = '.'.join(song.split('.')[:-1]) + '.bpm'
        full_doc_name = os.path.join(tempo_anno_dir, doc_name)
        with open(full_doc_name, 'r') as f:
            tempo = f.read()

        res.append(
            {
                "filename": path,
                # "samplerate": sr,
                "duration": len(wav) / sr,
                "tempo mean": tempo
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