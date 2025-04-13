import os
import json
import h5py
import numpy as np


def format_duration(t):
    t = round(t)
    sec = t % 60
    min = (t // 60) % 60
    return "{:02d}:{:02d}".format(int(min), int(sec))


def sec2token(st):
    x = st // 9
    r = st - 9 * x
    return x, round(32 / 10. * r)


def get_seg(x, st, ed):
    token_st, token_st_remain = sec2token(st)
    token_ed, token_ed_remain = sec2token(ed)
    token_dur, token_dur_remain = sec2token(ed - st)
    if token_ed_remain > 0:
        token_ed += 1
    # print(x.shape)
    # print(st, token_st, token_st_remain)
    # print(ed, token_ed, token_ed_remain)
    # print(ed - st, token_dur, token_dur_remain)
    x = x[int(token_st): int(token_ed)].reshape(-1, 768)
    return x[int(token_st_remain): int(token_dur * 32 + token_dur_remain + token_st_remain)]


def filter_data(data):
    new_data = []
    for d in data:
        flag = False
        if "segments" in d:
            for x in d["segments"]:
                if "key mode" in x and x["key mode"] not in ["none", "", "None"]:
                    flag = True
                    break
        elif "key mode" in d and len(d["key mode"].split(" - "))<2:
            flag = True
        if flag:
            new_data.append(d)
    return new_data


def split_dataset(root_folder, output_folder):
    splits = {}
    for dataset in os.listdir(root_folder):
        dataset_folder = os.path.join(root_folder, dataset)
        metadata = os.path.join(dataset_folder, "metadata.json")
        with open(metadata, "r") as f:
            data = json.load(f)
        for d in data:
            d["dataset"] = dataset

        splits[dataset] = data

    ratio = 0.9
    train = []
    test = []
    for dataset in splits:
        data = filter_data(splits[dataset])
        np.random.shuffle(data)
        data_len = len(data)

        training_num = int(data_len * ratio)
        print(dataset, data_len, training_num)
        if training_num > 0:
            train += data[:training_num]
            if training_num < data_len:
                test += data[training_num:]

    os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, "train.json"), "w") as f:
        json.dump(train, f, indent=2)

    with open(os.path.join(output_folder, "test.json"), "w") as f:
        json.dump(test, f, indent=2)


def load_dataset(dataset_path, feature_folder):
    with open(dataset_path, "r") as f:
        data = json.load(f)

    data_list = []
    datasets = []

    for d in data:
        if d["dataset"] not in datasets:
            datasets.append(d["dataset"])
        if "segments" in d:
            for seg in d["segments"]:
                seg["filename"] = d["filename"]
                seg["dataset"] = d["dataset"]
                seg["duration"] = float(seg["offset"]) - float(seg["onset"])
                data_list.append(seg)
        else:
            d["onset"] = 0
            d["offset"] = d["duration"]
            data_list.append(d)

    feature = {dataset: h5py.File(os.path.join(feature_folder, dataset + ".h5"), "r")
               for dataset in datasets}

    return data_list, feature


def contrast_tempo(sample_a, sample_b):
    val_1 = round(float(sample_a[1]["tempo mean"]))
    val_2 = round(float(sample_b[1]["tempo mean"]))
    dt = val_2 * 0.02
    if np.abs(val_1 - val_2) <= dt:
        desc = "comparable"
    elif np.abs(val_1 / 2 - val_2) <= dt:
        desc = "double speed"
    elif np.abs(val_1 * 2 - val_2) <= dt:
        desc = "half speed"
    elif np.abs(val_1 / 3 - val_2) <= dt:
        desc = "triple speed"
    elif np.abs(val_1 * 3 - val_2) <= dt:
        desc = "one-third speed"
    elif val_1 < val_2:
        desc = "slower"
    else:
        desc = "faster"
    return f"<tempo>{desc}</tempo>"


def caption_tempo(sample, noise, rng):
    val = round(float(sample["tempo mean"]))
    if noise:
        dt = round(val * 0.02)
        val = val + rng.randint(-dt, dt + 1)
    return f"<tempo>{val} bpm</tempo>"


def contrast_key(sample_a, sample_b):
    val_1 = sample_a[1]["key mode"]
    val_2 = sample_b[1]["key mode"]
    replace_dict = {":": "", "major": "maj", "minor": "min"}
    for name in replace_dict:
        val_1 = str.replace(val_1, name, replace_dict[name])
        val_2 = str.replace(val_2, name, replace_dict[name])

    if val_1 == val_2:
        desc = "same"
    elif val_1[-3:] == val_2[-3:]:
        desc = "same mode but different scale"
    elif val_1[:-3] == val_2[:-3]:
        desc = "same scale but different mode"
    else:
        desc = "different"
    return f"<key>{desc}</key>"


def caption_key(sample):
    replace_dict = {":": "", "major": "maj", "minor": "min"}
    val = sample['key mode']
    for name in replace_dict:
        val = str.replace(val, name, replace_dict[name])
    val = str.replace(val, "maj", " major")
    val = str.replace(val, "min", " minor")
    return f"<key>{val}</key>"


def caption_time_signature(sample):
    return f"<time-signature>{sample['time signature']}</time-signature>"


def exist_key(x):
    return "key mode" in x and len(x["key mode"].split(" - ")) == 1


def exist_time_signature(x):
    return "time signature" in x and len(x["time signature"].split(" - ")) == 1


def wrap_qa(res, tag, tab, eos):
    q = f"<{tab} {tag}>"
    if len(res) > 1:
        q += ''.join(res[:-1])
    last = res[-1].split("><")
    q = q + last[0] + "><"
    a = "><".join(last[1:]) + f"</{tab}>{eos}"
    return q, a


def create_contrast(groups, contrast_tuples, eos, attributes, inference=False):
    res = []

    tag = " ".join([f"({chr(ord('A') + a)}, {chr(ord('A') + b)})" for a, b in contrast_tuples])
    for a, b in contrast_tuples:
        desc = ""
        attrs = []
        if "tempo" in attributes:
            desc += contrast_tempo(groups[a], groups[b])
            attrs.append("tempo")
        if "key" in attributes and exist_key(groups[a][1]) and exist_key(groups[b][1]):
            desc += contrast_key(groups[a], groups[b])
            attrs.append("key")
        tuple = f"({chr(ord('A') + a)}, {chr(ord('A') + b)})"
        res.append(f"<{tuple} {' '.join(attrs)}>{desc}</{tuple}>")

    return wrap_qa(res, tag, "contrast", eos)


def create_caption(groups, eos, attributes, rng, inference):
    res = []
    tags = []
    for tag, data in groups:
        descs = []
        if "tempo" in attributes and "tempo mean" in data:
            descs.append(
                {"desc": caption_tempo(data, not inference, rng=rng),
                 "key": "tempo"})
        if "key" in attributes and exist_key(data):
            descs.append(
                {"desc": caption_key(data),
                 "key": "key"})
        if "time-signature" in attributes and exist_time_signature(data):
            descs.append(
                {"desc": caption_time_signature(data),
                 "key": "time-signature"})
        if not inference:
            rng.shuffle(descs)
        props = ' '.join([d['key'] for d in descs])
        desc = ''.join([d['desc'] for d in descs])
        res.append(f"<{tag} {props}>{desc}</{tag}>")
        tags.append(tag)
    return wrap_qa(res, ' '.join(tags), "analysis", eos)


def sample_contrast_tuple(groups, rng):
    pairs = [[i, j] for i in range(len(groups)) for j in range(i + 1, len(groups))]
    if len(pairs) == 0:
        pairs = [[0, 0]]
    n_pairs = rng.randint(1, min(len(pairs) + 1, 11))
    rng.shuffle(pairs)
    return pairs[:n_pairs]


def create_qa(q, a, inference=False):
    head_desc = q if inference else ""
    cap_desc = a if inference else q + a
    return head_desc, cap_desc


def create_desc(data, feature, eos, eot, rng, attributes, inference):
    tags = []
    head = []
    embs = []
    groups = []
    for i in range(len(data)):
        tag = chr(ord('A') + i)
        st = float(data[i]["onset"])
        ed = float(data[i]["offset"])
        dur = format_duration(float(data[i]["duration"]))
        filename = data[i]["filename"]
        emb = get_seg(feature[data[i]["dataset"]][filename], st, ed)
        emb_tag = "".join(["<|x|>"] * len(emb))
        embs.append(emb)
        desc = f"<{tag} duration feature><duration>{dur}</duration><feature>{emb_tag}</feature></{tag}>"
        head.append(desc)
        tags.append(tag)
        groups.append([tag, data[i]])
    head = f"<Audio {' '.join(tags)}>{''.join(head)}</Audio>{eot}"
    flag = len(groups) > 1 and rng.rand() > 0.5
    head_desc, caps_desc, contrast_head_desc, contrast_caps_desc = None, None, None, None
    if (flag or inference) and inference:
        contrast_tuples = sample_contrast_tuple(groups, rng) if not inference else [[0, 1]]
        q, a = create_contrast(groups=groups,
                               contrast_tuples=contrast_tuples,
                               eos=eos, attributes=attributes, inference=inference)
        contrast_head_desc, contrast_caps_desc = create_qa(q=q,
                                                           a=a,
                                                           inference=inference)
        contrast_head_desc = head + contrast_head_desc
    if not flag or inference or True:
        q, a = create_caption(groups, eos=eos, attributes=attributes, rng=rng, inference=inference)
        head_desc, caps_desc = create_qa(q=q,
                                         a=a,
                                         inference=inference)
        head_desc = head + head_desc

    return head_desc, caps_desc, contrast_head_desc, contrast_caps_desc, embs
