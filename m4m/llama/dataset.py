import os

import h5py
from torch.utils.data import Dataset as BaseDataset
from ..dataset.dataset_generator.create_dataset import create_caption

import numpy as np
import json

A_CONTENT = 128256
MAX_SEQ = 2048 + 1
FEATURE_DIM = 768
MAX_POS = int(18*75 + 1)


def load_data(dataset_path):
    with open(dataset_path, "r") as f:
        data = json.load(f)
    return data


def load_feature(feature_folder):
    feature = {}
    for dataset in os.listdir(feature_folder):
        path = os.path.join(feature_folder, dataset)
        feature[dataset.split(".h5")[0]] = h5py.File(path, "r")
    return feature


class MusicDataset(BaseDataset):
    def __init__(self, tokenizer, data_path, feature_folder, inference=False, validation=False):
        super().__init__()
        self.tokenizer = tokenizer
        print('data_path:', data_path)
        self.data = load_data(data_path)
        self.split = data_path.split('/')[-1].split(".json")[0]
        self.rng = np.random.RandomState(4321) if inference else np.random.RandomState(np.random.randint(0, 1234))
        self.feature = load_feature(feature_folder)
        self.eot = "<|eot_id|>"
        self.eos = "<|end_of_text|>"
        self.training_samples = self.regenerate_training_samples(not inference)
        print("init", len(self.training_samples))
        print('inference status', inference)
        self.validation = validation
        self.init = True

    def regenerate_training_samples(self, drop_out):        
        if '3000' in self.split:
            print(f'using already existing file {self.split}')
            with open(f'dataset/new_dataset/formatted_dataset/caption_{self.split}.json', 'r') as f:
                data = json.load(f)
            print(f'dataset/new_dataset/formatted_dataset/caption_{self.split}.json')        

        else:
            print('using create_caption')
            data = create_caption(None, None,
                                training_data=self.data, split=self.split, rng=self.rng,
                                eos=self.eos, eot=self.eot, feature_token="<|x|>",
                                drop_out=drop_out, overlapping_ratio=1,
                                save_dict=False, fps=50)
        self.rng.shuffle(data)
        return data

    def __len__(self):
        return len(self.training_samples)

    def inference(self):
        for i in range(self.__len__()):
            tokens = self.__getitem__(i, inference=True)
            yield {
                "Q": tokens["Q"],
                "A": tokens["A"],
                "clap_rep": tokens["clap_rep"],
                "pos_id": tokens["pos_id"],
                "input_ids": tokens["input_ids"],
                "filename": tokens["filename"]
            }

    def wrap_tokens(self, head, caps, feature, inference):
        question_tokens = self.tokenizer(head)
        tokens = self.tokenizer(head + caps) if not inference else question_tokens
        input_ids = tokens["input_ids"]   
        input_ids = np.array(input_ids)

        if len(input_ids) > MAX_SEQ:
            input_ids = input_ids[:MAX_SEQ]

        audio_pos = np.array(input_ids) == A_CONTENT
        n = int(audio_pos.sum())

        if n != len(feature): # n == len(feature) + 1
            new_shape = (n, 768)
            padded_features = np.zeros(new_shape)
            padded_features[:len(feature), :] = feature
            feature = padded_features

        assert n == len(feature)
        pos_id = np.zeros([MAX_POS], dtype=np.int16)
        pos_id[:len(feature)] = 1
        feature_tokens = np.zeros([MAX_POS, FEATURE_DIM], dtype=np.float32)
        feature_tokens[:len(feature)] = feature

        tokens["clap_rep"] = feature_tokens
        tokens["pos_id"] = pos_id
        if not inference:
            loss_mask = np.zeros([MAX_SEQ])
            loss_mask[len(question_tokens["input_ids"]): len(input_ids)] = 1
            tokens["loss_mask"] = loss_mask
            return tokens
        return tokens

    def wrap_tokens_single(self, head, caps, feature, inference):
        question_tokens = self.tokenizer(head)
        tokens = self.tokenizer(head + caps) if not inference else question_tokens
        input_ids = tokens["input_ids"]
        input_ids = np.array(input_ids)
        if len(input_ids) > MAX_SEQ:
            input_ids = input_ids[:MAX_SEQ]
        audio_pos = np.array(input_ids) == A_CONTENT
        n = int(audio_pos.sum())
        if n != len(feature):
            new_shape = (n, 768)
            padded_features = np.zeros(new_shape)
            padded_features[:len(feature), :] = feature
            feature = padded_features
        pos_id = np.zeros([MAX_POS], dtype=np.int16)
        pos_id[:len(feature)] = 1
        feature_tokens = np.zeros([MAX_POS, FEATURE_DIM], dtype=np.float32)
        feature_tokens[:len(feature)] = feature
        tokens["clap_rep"] = feature_tokens
        tokens["pos_id"] = pos_id
        if not inference:
            loss_mask = np.zeros([MAX_SEQ])
            loss_mask[len(question_tokens["input_ids"]): len(input_ids)] = 1
            tokens["loss_mask"] = loss_mask
            return tokens
        return tokens

    def __getitem__(self, idx, inference=False):

        if idx >= self.__len__():
            self.init = False
            self.training_samples = self.regenerate_training_samples(drop_out=True)
            raise StopIteration

        if self.init and not inference and not self.validation:
            tokens = {
                "input_ids": []
            }
            return tokens

        training_sample = self.training_samples[idx]
        desc = training_sample["caption"]
        filename = training_sample["filename"]
        dataset = training_sample["dataset"]
        n_tokens_st = training_sample["n_tokens_st"]
        n_tokens_ed = training_sample["n_tokens_ed"]
        if filename not in self.feature[dataset]:
            return self.__getitem__(idx + 1, inference=inference)
        feature = self.feature[dataset][filename][n_tokens_st: n_tokens_ed]


        head, caps = desc.split(self.eot)
        head = head + self.eot

        data = self.wrap_tokens(head, caps, feature, inference)

        if inference:
            data["Q"] = head
            data["A"] = caps
            data["filename"] = filename
        return data
