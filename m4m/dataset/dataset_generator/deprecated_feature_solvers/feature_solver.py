import os
import h5py as h5
import numpy as np
import torch
import librosa
import json
from ...utils import get_device
from ...audiocraft.models.loaders import load_compression_model
from demucs.audio import convert_audio

import sys

# sys.path.insert(0, os.path.join(sys.path[0], "../../.."))


cache_dir = os.environ.get('MUSICGEN_ROOT', None)
name = None
device = get_device()
compression_model = load_compression_model("large", device=device, cache_dir=cache_dir)
compression_model.eval()
sample_rate = 32000


def extract_rvq(audio_path):
    x, sr = librosa.load(audio_path, mono=True)
    x = torch.from_numpy(x[None, ...])
    x = convert_audio(x, sr, sample_rate, 1).unsqueeze(0)
    print(audio_path)
    with torch.no_grad():
        codes, _ = compression_model.encode(x.to(device))
        emb = compression_model.quantizer.decode(codes)
        print(emb.shape[-1] / (x.shape[-1] / sample_rate), emb.shape)
    return emb.squeeze(0).transpose(0, 1).cpu().numpy()


def save_feature(metadata_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    hfs = {}
    for dataset in os.listdir(metadata_folder):
        # if 'AAM' in dataset or 'CBF' in dataset or 'CCOM' in dataset or 'FSLD' in dataset:
        #     continue
        # if 'guitarset' in dataset or 'IRMAS' in dataset or 'jaCappella' in dataset:
        #     continue
        
        # if 'CCOM' not in dataset:
        #     continue
        if 'FMA' not in dataset:
            continue
        print(dataset)
        file_path = os.path.join(metadata_folder, dataset, "metadata.json")
        with open(file_path, "r") as f:
            data = json.load(f)
        path = os.path.join(output_folder, dataset + ".h5")

        hfs[dataset] = h5.File(path, "a")

        with open('/data2/ruihan/debug/all_m4m/FMA_notyet.txt', 'a') as f:
            f.write('=====================\n')

        for d in data:
            filename = d["filename"]
            if filename in hfs[dataset]:
                continue
            try:
                feature = extract_rvq(filename)
            except:
                print("Error extracting: ", filename)
                with open('/data2/ruihan/debug/all_m4m/FMA_notyet.txt', 'a') as f:
                    f.write(filename + '\n')
                continue

            dset = hfs[dataset].create_dataset(filename, feature.shape, dtype="float32")
            dset[:] = feature

        hfs[dataset].close()
