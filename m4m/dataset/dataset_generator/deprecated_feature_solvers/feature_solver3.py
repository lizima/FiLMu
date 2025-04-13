import os
import h5py as h5
import numpy as np
import torch
import librosa
import json
from ...utils import get_device
# from ...audiocraft.models.loaders import load_compression_model
from demucs.audio import convert_audio
# from transformers import AutoModel
# from transformers import Wav2Vec2FeatureExtractor
# import torchaudio.transforms as T
import laion_clap


import sys

cache_dir = os.environ.get('MUSICGEN_ROOT', None)
name = None
device = get_device()
# compression_model = load_compression_model("large", device=device, cache_dir=cache_dir)
# compression_model.eval()
sample_rate = 32000

clap_model = laion_clap.CLAP_Module(enable_fusion=False)
clap_model.load_ckpt()

def cut_audio(x):
    ls = []
    return ls

def extract_rvq3(audio_path):
    x, sr = librosa.load(audio_path, mono=True)
    
    x = torch.from_numpy(x[None, ...])
    x = convert_audio(x, sr, sample_rate, 1)
    # x = x.cpu().numpy()
    # print(x.shape, 'x shape')
    print(audio_path)
    print(x.shape[1]/sample_rate, 'seconds')

    # audio_clips = cut_audio(x)
    audio_clips = [x]
    for audio in audio_clips:
        with torch.no_grad():
            emb = clap_model.get_audio_embedding_from_data(x = audio, use_tensor=True)
            emb = emb.unsqueeze(2)
            print(emb.shape)
        # assert(0)
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
        if 'adc' not in dataset:
            continue

        print(dataset)
        file_path = os.path.join(metadata_folder, dataset, "metadata.json")
        
        with open(file_path, "r") as f:
            data = json.load(f)
        path = os.path.join(output_folder, dataset + ".h5")

        hfs[dataset] = h5.File(path, "a")
        for d in data:
            filename = d["filename"]

            if filename in hfs[dataset]:
                continue

            feature = extract_rvq3(filename)
            dset = hfs[dataset].create_dataset(filename, feature.shape, dtype="float32")
            dset[:] = feature

        hfs[dataset].close()