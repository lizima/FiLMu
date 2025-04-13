import os
import json
import librosa

dic_root = {
    'Ab': 'G#',
    'Bb': 'A#',
    'Cb': 'B',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
}

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
    root_folder_ = root_folder + '/giantsteps-key-dataset/audio'
    res = []
    for song in os.listdir(root_folder_):
        path = os.path.join(root_folder_, song)
        wav, sr = librosa.load(path)

        key_anno_dir = root_folder + '/giantsteps-key-dataset/annotations/key'

        doc_name = '.'.join(song.split('.')[:-1]) + '.key'
        full_doc_name = os.path.join(key_anno_dir, doc_name)
        with open(full_doc_name, 'r') as f:
            key = f.read()
        
        if 'b' in key:
            print('original key:', key)
            root = key.split('b')[0] + 'b'
            scale = key.split('b')[1]
            key = dic_root[root] + scale
            print('new key:', key)

        res.append(
            {
                "filename": path,
                # "samplerate": sr,
                "duration": len(wav) / sr,
                "key mode": key
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