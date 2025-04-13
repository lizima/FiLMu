import os
import h5py as h5
import numpy as np
import torch
import librosa
import json
from ...utils import get_device
from demucs.audio import convert_audio
from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor
import torchaudio.transforms as T


import sys

device = get_device()

sample_rate = 32000



def extract_rvq2(audio_path, processor, resampler, mert_model):

    x, sr = librosa.load(audio_path, mono=True)
    x = torch.from_numpy(x[None, ...])
    x = convert_audio(x, sr, sample_rate, 1).squeeze(0)
    input_audio = resampler(x)
    inputs = processor(input_audio, sampling_rate=processor.sampling_rate, return_tensors="pt")
    
    print(audio_path)
    with torch.no_grad():
        inputs = inputs.to(get_device())
        output = mert_model(**inputs, output_hidden_states=True)
        emb = output.hidden_states[-1]
        emb = emb.transpose(1, 2)
        print(emb.shape[-1] / (x.shape[-1] / sample_rate), emb.shape)
    return emb.squeeze(0).transpose(0, 1).cpu().numpy()


def save_feature(metadata_folder, output_folder, suffix='', selected_dataset=None, re_extract = False):
    os.makedirs(output_folder, exist_ok=True)
    log_dir = os.path.join(output_folder, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    hfs = {}
    for dataset in os.listdir(metadata_folder):
        print(dataset)

        if selected_dataset is None:
            a = 0
        elif selected_dataset == 'FMA':
            if f'FMA_{suffix}' not in dataset:
                continue
        elif selected_dataset == 'MTG':
            if f'MTG_{suffix}' not in dataset:
                continue
        else:
            if selected_dataset not in dataset:
                continue

        processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True, local_files_only=True)
        resample_rate = processor.sampling_rate
        resampler = T.Resample(sample_rate, resample_rate)
        mert_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True, local_files_only=True)
        mert_model = mert_model.to(get_device())
        print('dataset:', dataset)
        file_path = os.path.join(metadata_folder, dataset, "metadata.json")
        print('file_path:', file_path)
        
        with open(file_path, "r") as f:
            data = json.load(f)
        path = os.path.join(output_folder, dataset + ".h5")
        print('path:', path)

        hfs[dataset] = h5.File(path, "a")

        if re_extract:

            not_extracted_files = []
            with open(f'{log_dir}/feature_{dataset}_{suffix}_r{re_extract}.txt', 'r') as f:
                not_extracted_files = f.readlines()
                not_extracted_files = [f.strip() for f in not_extracted_files]

            for d in data:
                filename = d["filename"]

                if filename in hfs[dataset]:
                    continue
                
                if filename not in not_extracted_files:
                    continue
                
                print('filename:', filename)
                try:
                    feature = extract_rvq2(filename, processor, resampler, mert_model)
                    dset = hfs[dataset].create_dataset(filename, feature.shape, dtype="float32")
                    dset[:] = feature
                except:
                    with open(f'{log_dir}/feature_{dataset}_{suffix}_r{int(re_extract)+1}.txt', 'a') as f:
                        f.write(filename)
                        f.write('\n')
        else:
            for d in data:
                filename = d["filename"]
                if filename in hfs[dataset]:
                    continue
                print('filename:', filename)
                try:
                    feature = extract_rvq2(filename, processor, resampler, mert_model)
                    dset = hfs[dataset].create_dataset(filename, feature.shape, dtype="float32")
                    dset[:] = feature
                except:
                    with open(f'{log_dir}/feature_{dataset}_{suffix}_r0.txt', 'a') as f:
                        f.write(filename)
                        f.write('\n')

        hfs[dataset].close()