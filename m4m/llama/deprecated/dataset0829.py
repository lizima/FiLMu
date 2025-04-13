import os

import h5py
from torch.utils.data import Dataset as BaseDataset
from ..dataset.dataset_generator.create_dataset import create_caption
from ..dataset.dataset_generator.create_dataset import create_caption4
from ..dataset.dataset_generator.create_dataset import create_caption4_
from ..dataset.dataset_generator.create_dataset import create_caption5_
from ..dataset.dataset_generator.create_dataset import create_caption6_
from ..dataset.dataset_generator.create_dataset import create_caption7_
from ..dataset.dataset_generator.create_dataset import create_caption8_
from ..dataset.dataset_generator.create_dataset import create_caption9_
from ..dataset.dataset_generator.create_dataset import create_caption10_
from ..dataset.dataset_generator.create_dataset import create_caption11_
from ..dataset.dataset_generator.create_dataset import create_caption12_
from ..dataset.dataset_generator.create_dataset import create_caption300

import numpy as np
import json

A_CONTENT = 128256
MAX_SEQ = 2048 + 1
# FEATURE_DIM = 128
FEATURE_DIM = 768
# MAX_POS = int(32*50 + 1)
MAX_POS = int(22*75 + 1)


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
    def __init__(self, tokenizer, data_path, feature_folder, inference=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.data = load_data(data_path)
        self.split = data_path.split('/')[-1].split(".json")[0]
        self.rng = np.random.RandomState(4321) if inference else np.random.RandomState(np.random.randint(0, 1234))
        self.feature = load_feature(feature_folder)
        self.eot = "<|eot_id|>"
        self.eos = "<|end_of_text|>"
        self.training_samples = self.regenerate_training_samples(not inference)
        print("init", len(self.training_samples))
        print('inference status', inference)
        self.init = True

    def regenerate_training_samples(self, drop_out):
        if '000' in self.split:
            print('using create_caption4')
            data = create_caption4(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=50)
        elif '500' in self.split:
            print('using create_caption4_')
            data = create_caption4_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)

        elif '501' in self.split:
            print('using create_caption5_')
            data = create_caption5_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75) 
            
        elif '502' in self.split:
            print('using create_caption6_')
            data = create_caption6_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)    
        elif '503' in self.split:
            print('using create_caption7_')
            data = create_caption7_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)

        elif '504' in self.split:
            print('using create_caption8_')
            data = create_caption8_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)

        elif '150' in self.split:
            print('using create_caption9_')
            data = create_caption9_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)

        elif '151' in self.split:
            print('using create_caption10_')
            data = create_caption10_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)
        elif '152' in self.split:
            print('using create_caption11_')
            data = create_caption11_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)          

        elif '250' in self.split or '251' in self.split or '252' in self.split or '222' in self.split:
            print('using create_caption12_')
            data = create_caption12_(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=75)                  
            
        elif '300' in self.split or '301' in self.split or '401' in self.split or '402' in self.split or '403' in self.split or '404' in self.split:
            print('using create_caption300')
            data = create_caption300(None, None,
                        training_data=self.data, split=self.split, rng=self.rng,
                        eos=self.eos, eot=self.eot, feature_token="<|x|>",
                        drop_out=drop_out, overlapping_ratio=1,
                        save_dict=False, fps=50)
        
        elif '700' in self.split:
            print('using create_caption [700]')
            data = create_caption(None, None,
                                training_data=self.data, split=self.split, rng=self.rng,
                                eos=self.eos, eot=self.eot, feature_token="<|x|>",
                                drop_out=drop_out, overlapping_ratio=1,
                                save_dict=False, fps=75, with_comparison=True)
        elif '701' in self.split:
            print('using create_caption [701]')
            data = create_caption(None, None,
                                training_data=self.data, split=self.split, rng=self.rng,
                                eos=self.eos, eot=self.eot, feature_token="<|x|>",
                                drop_out=drop_out, overlapping_ratio=1,
                                save_dict=False, fps=75, with_comparison=False)
            
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
        # print('ThisIsTest inference')
        for i in range(self.__len__()):
            # print('ThisIsTest iiiiinnnn')
            tokens = self.__getitem__(i, inference=True)
            # print('ThisIsTest tokens["Q"][0:5]', tokens["Q"][0:5])
            yield {
                "Q": tokens["Q"],
                "A": tokens["A"],
                "clap_rep": tokens["clap_rep"],
                "pos_id": tokens["pos_id"],
                "input_ids": tokens["input_ids"],
            }

    def wrap_tokens(self, head, caps, feature, inference):
        question_tokens = self.tokenizer(head)
        # print('ThisIsTest wrap_tokens', inference)
        tokens = self.tokenizer(head + caps) if not inference else question_tokens
        # print('ThisIsTest tokens_shape', tokens.shape) # not valid
        # print('ThisIsTest tokens', tokens[0:5])
        input_ids = tokens["input_ids"]
        # print('ThisIsTest input_ids_shape', input_ids.shape) # not valid
        # print('ThisIsTest input_ids', input_ids[0:5])        
        input_ids = np.array(input_ids)
        # print('input_ids1', input_ids)

        if len(input_ids) > MAX_SEQ:
            input_ids = input_ids[:MAX_SEQ]
        # print('input_ids2', input_ids)
        audio_pos = np.array(input_ids) == A_CONTENT
        # print('audio_pos', audio_pos)
        n = int(audio_pos.sum())
        # print('n', n)
        # print('len(feature)', len(feature))

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

    def __getitem__(self, idx, inference=False):

        if idx >= self.__len__():
            self.init = False
            self.training_samples = self.regenerate_training_samples(drop_out=True)
            raise StopIteration

        if self.init and not inference:
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
        feature = self.feature[dataset][filename][n_tokens_st: n_tokens_ed]

        # print("mark:", self.feature[dataset][filename].shape)
        # print('dataset:', dataset)
        # print('filename:', filename)
        # print('n_tokens_st:', n_tokens_st)
        # print('n_tokens_ed:', n_tokens_ed)

        head, caps = desc.split(self.eot)
        head = head + self.eot
        # print('head:', head)
        # print('caps:', caps)
        data = self.wrap_tokens(head, caps, feature, inference)
        # print('data:', data)
        # print('input_ids', len(data['input_ids']))
        # print('attention_masks', len(data['attention_mask']))
        # print('clap_rep', data['clap_rep'].shape)
        # input_ids 1730                                                                                        
        # attention_masks 1730                                                                                  
        # clap_rep (1601, 128) # 这个rep换成别的维度的到底行不行呢

        if inference:
            data["Q"] = head
            data["A"] = caps
        return data

    # def map(self, tokenize_row, num_proc):
    #     print(tokenize_row)
    #     print(num_proc)
    #     return self

    # def data_collator(self, batch):
    #     device = self.embeddings.device
    #     input_ids = torch.from_numpy(np.stack([d["input_ids"] for d in batch], 0))
    #     attention_mask = torch.from_numpy(np.stack([d["attention_mask"] for d in batch], 0))
    #     return {
    #         "input_ids": input_ids,
    #         "attention_mask": attention_mask
    #     }

    # <A-CONTENT> 32001
    # <A-HYPHEN> 32002
