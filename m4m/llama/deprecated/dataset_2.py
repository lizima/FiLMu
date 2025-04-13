from torch.utils.data import Dataset as BaseDataset
from .data_generator_2 import load_dataset, create_desc
import numpy as np
import os
import h5py

A_CONTENT = 128256
MAX_SEQ = 480
FEATURE_DIM = 768
MAX_DUR = 360
MAX_N_SAMPLES = 1
MAX_POS = int(MAX_DUR / 9) * 32 + MAX_N_SAMPLES * 32



class MusicDataset(BaseDataset):
    def __init__(self, tokenizer, data_path, feature_folder, attributes, inference=False):
        super().__init__()
        self.tokenizer = tokenizer
        data_list, feature = load_dataset(data_path, feature_folder)
        self.rng = np.random.RandomState(4321) if inference else np.random.RandomState(np.random.randint(0, 1234))
        self.rng.shuffle(data_list)
        self.feature = feature
        self.data_list = data_list
        self.eot = "<|eot_id|>"
        self.eos = "<|end_of_text|>"
        self.attributes = attributes
        print("init", len(self.data_list))
        self.init = True

    def __len__(self):
        return len(self.data_list)

    def inference(self):
        for i in range(self.__len__()):
            tokens, Q, A, con_tokens, cQ, cA = self.__getitem__(i, inference=True)
            yield [{
                "Q": Q,
                "A": A,
                "clap_rep": tokens["clap_rep"],
                "pos_id": tokens["pos_id"],
                "input_ids": tokens["input_ids"],
            }, {
                "Q": cQ,
                "A": cA,
                "clap_rep": con_tokens["clap_rep"],
                "pos_id": con_tokens["pos_id"],
                "input_ids": con_tokens["input_ids"],
            }]

    def wrap_tokens(self, head, caps, feature, inference):
        question_tokens = self.tokenizer(head)
        tokens = self.tokenizer(head + caps) if not inference else question_tokens
        input_ids = tokens["input_ids"]
        input_ids = np.array(input_ids)

        audio_pos = np.array(input_ids) == A_CONTENT
        n = int(audio_pos.sum())
        feature = np.concatenate([emb.reshape(-1, FEATURE_DIM) for emb in feature], 0)
        assert n == len(feature)
        pos_id = np.zeros([MAX_POS], dtype=np.int16)
        pos_id[:len(feature)] = 1
        clap_rep = np.zeros([MAX_POS, FEATURE_DIM], dtype=np.float32)
        clap_rep[:len(feature)] = feature

        tokens["clap_rep"] = clap_rep
        tokens["pos_id"] = pos_id
        if not inference:
            loss_mask = np.zeros([2048])
            loss_mask[len(question_tokens["input_ids"]): len(input_ids)] = 1
            tokens["loss_mask"] = loss_mask

        return tokens

    def __getitem__(self, idx, inference=False):

        if idx >= self.__len__():
            self.init = False
            raise StopIteration
        if self.init and not inference:
            tokens = {
                "input_ids": []
            }
            return tokens
        rng = self.rng

        data = [self.data_list[idx]]
        max_dur = MAX_DUR - int(data[0]["duration"])
        n_data = rng.randint(0, MAX_N_SAMPLES) if max_dur > 0 else 0
        if inference:
            n_data = 1
        n_attemps = 5 if not inference else 20
        if max_dur < 0:
            n_data += 1
            max_dur = MAX_DUR
            data = []
        while n_data > 0 and n_attemps > 0:
            sampled_data = self.data_list[rng.randint(0, len(self.data_list))]
            if max_dur - int(sampled_data["duration"]) < 0:
                if len(data) > 0:
                    n_attemps -= 1
                continue
            n_data -= 1
            max_dur -= int(sampled_data["duration"])
            data.append(sampled_data)
        data = data[1:] + [data[0]]
        head_desc, caps_desc, contrast_head_desc, contrast_caps_desc, feature = create_desc(data, self.feature,
                                                                                            eot=self.eot,
                                                                                            eos=self.eos, rng=self.rng,
                                                                                            attributes=self.attributes,
                                                                                            inference=inference)

        if head_desc is not None:
            tokens = self.wrap_tokens(head_desc, caps_desc, feature, inference)
            if not inference:
                return tokens

        if contrast_head_desc is not None:
            contrast_tokens = self.wrap_tokens(contrast_head_desc, contrast_caps_desc, feature, inference)
            if not inference:
                return contrast_tokens

        return tokens, head_desc, caps_desc, contrast_tokens, contrast_head_desc, contrast_caps_desc

    def map(self, tokenize_row, num_proc):
        print(tokenize_row)
        print(num_proc)
        return self

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
