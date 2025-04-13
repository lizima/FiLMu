import json
import os
import numpy as np
from .utils import format_props, crop_props, format_duration, divide_into_more_segs, crop_props_revise, get_pairs, get_comparison
import re
# from .utils import format_props_revise, format_props2_revise


def filter_data(data, key_words=None):
    new_data = []
    for d in data:
        flag = False
        for x in d["segments"]:
            if key_words is None:
                flag = True
                break
            else:
                for key in key_words:
                    if key in x and x[key] not in ["none", "", "None"]:
                        flag = True
                        break

        if flag:
            new_data.append(d)
    return new_data


def split_dataset(root_folder, output_folder, suffix = "", selected_datasets=None):
    splits = {}
    for dataset in os.listdir(root_folder):
        if dataset not in selected_datasets:
            continue
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
        if training_num > 0:
            train += data[:training_num]
            if training_num < data_len:
                test += data[training_num:]

    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    with open(os.path.join(output_folder, f"train{suffix}.json"), "w") as f:
        json.dump(train, f, indent=2)

    with open(os.path.join(output_folder, f"test{suffix}.json"), "w") as f:
        json.dump(test, f, indent=2)

    with open(os.path.join(output_folder, f"valid{suffix}.json"), "w") as f:
        json.dump(test[0:50], f, indent=2)


def segs2caption(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps):

    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100


    tag_idx = 0
    marks = []
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")
        tag = f"<timestamp>{timestamp}</timestamp>"
        tag += "".join(
            [f"<{k}>{crop_props(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
             for k in keys if offset - float(seg['onset']) > 0.5])
        out_keys = ["timestamp"] + keys
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1
    desc = f"<music {' '.join(marks)}>{props}</music>"
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    dur = format_duration(max_offset - onset)
    n_tokens_st = int(onset * fps)
    n_tokens_ed = min(max_n_tokens, int(max_offset * fps)) if max_n_tokens is not None else int(max_offset * fps)
    feature = "".join([feature_token] * (n_tokens_ed - n_tokens_st))
    head = f"<audio duration feature><duration>{dur}</duration><feature>{feature}</feature></audio>"
    return head + desc + eos, n_tokens_st, n_tokens_ed, onset, max_offset

def segs2caption_revise(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps, with_comparison=False):

    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100


    tag_idx = 0
    marks = []
    comp_dic = {}
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    for key in keys:
        comp_dic[key] = {}
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")
        for key in keys:
            comp_dic[key][tag_idx] = seg[key]
        tag = f"<timestamp>{timestamp}</timestamp>"
        tag += "".join(
            [f"<{k}>{crop_props_revise(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
             for k in keys if offset - float(seg['onset']) > 0.5])

        out_keys = ["timestamp"] + keys
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1

    desc = f"<music {' '.join(marks)}>{props}</music>"

    # keys, 一个list
    comp = ""
    if tag_idx > 1:
        pairs = get_pairs(tag_idx, rng)
        if len(pairs) > 0:
            comp_1 = get_comparison(pairs, keys, comp_dic)
            comp = f"<comparison {' '.join(pairs)}>{comp_1}</comparison>"
            
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    dur = format_duration(max_offset - onset)
    n_tokens_st = int(onset * fps)
    n_tokens_ed = min(max_n_tokens, int(max_offset * fps)) if max_n_tokens is not None else int(max_offset * fps)
    # print('Start, End, dur', n_tokens_st, n_tokens_ed, dur)
    feature = "".join([feature_token] * (n_tokens_ed - n_tokens_st))
    head = f"<audio duration feature><duration>{dur}</duration><feature>{feature}</feature></audio>"
    if with_comparison:
        if with_comparison == 'only':
            # return head + f"<music {' '.join(marks)}></music>" + eot + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            if comp != "":
                first_part = comp_1.split('>')[0] + '>'
                second_part = '>'.join(comp_1.split('>')[1:])
                tmp = first_part + eot + second_part
                return head + f"<music {' '.join(marks)}></music>" + f"<comparison {' '.join(pairs)}>{tmp}</comparison>" + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            else:
                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset # reaturn anything, will drop afterwards
            
        elif with_comparison == 'only1':
            if comp != "":
                return head + f"<music {' '.join(marks)}></music>" + eot + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            else:
                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset # reaturn anything, will drop afterwards
        elif with_comparison == 'only2':
            return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
        elif with_comparison in ['only3', 'only4', 'only5', 'only6']:
            if comp == "":
                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
            else:
                if with_comparison == 'only3':
                    mask_rate = 0.1
                elif with_comparison == 'only4':
                    mask_rate = 0.2
                elif with_comparison == 'only5':
                    mask_rate = 0.5
                elif with_comparison == 'only6':
                    mask_rate = 1.0
                
                pattern = re.compile(r"<key>(.*?)</key>|<tempo>(.*?)</tempo>")
                # matches = pattern.findall(desc)
                matches_with_indices = []
                for match in re.finditer(pattern, desc):
                    matched_string = match.group(1) if match.group(1) else match.group(2)
                    start_idx = match.start(1) if match.group(1) else match.start(2)
                    matches_with_indices.append((matched_string, start_idx))

                idxs_to_replace = []
                for match, idx in matches_with_indices:
                    idxs_to_replace.append((idx, idx + len(match)))

                mask_idx = 0
                for tup in idxs_to_replace:
                    if rng.rand() < mask_rate:
                        # a = desc[tup[0]:tup[1]]
                        b = f'mask{mask_idx}' + '@'*(tup[1]-tup[0]-len(f'mask{mask_idx}'))
                        desc = desc[:tup[0]] + b + desc[tup[1]:]
                        mask_idx += 1

                desc = desc.replace("@", "")

                return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
        else:
            return head + desc + comp + eos, n_tokens_st, n_tokens_ed, onset, max_offset
    else:
        # print('without comparison')
        return head + desc + eos, n_tokens_st, n_tokens_ed, onset, max_offset

def segs2caption_revise_before(segs, onset, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps):

    props = ""
    offsets = []
    tempo_dt = rng.randint(0, 3) - 2 if drop_out else 0
    tempo_dt = tempo_dt * 1. / 100


    tag_idx = 0
    marks = []
    keys = [k for k in segs[0] if k not in ["mark", "onset", "offset", "timestamp"]]
    rng.shuffle(keys)

    if len(segs) == 1 and segs[0]["mark"] == "M":
        segs = divide_into_more_segs(segs[0], rng)

    if len(segs) > 1 and drop_out and rng.rand() > .8:
        drop_idx = rng.randint(0, len(segs))
    else:
        drop_idx = -1

    for i, seg in enumerate(segs):
        if float(seg["offset"]) - float(seg["onset"]) < 1:
            drop_idx = -1
            continue
        if drop_idx == i:
            continue
        offset = float(seg["offset"])

        offsets.append(float(seg["offset"]))
        # timestamp = format_duration(float(seg["onset"]) - onset) + "-" + format_duration(offset - onset)
        tmp1 = float(seg["onset"]) - onset
        tmp2 = offset - onset
        # timestamp = f'{tmp1:.1f}' + '-' + f'{tmp2:.1f}'
        timestamp = f'{tmp1:.1f}' + ', ' + f'{tmp2:.1f}'
    
        audio_tag = chr(tag_idx + ord('A'))
        marks.append(f"{audio_tag}({timestamp})")
        tag = f"<timestamp>({timestamp})</timestamp>"
        tag += "".join(
            [f"<{k}>{crop_props_revise(k, seg[k], float(seg['onset']), offset, onset, aug=drop_out, tempo_dt=tempo_dt)}</{k}>"
             for k in keys if offset - float(seg['onset']) > 0.5])

        out_keys = ["timestamp"] + keys
        if props == "":
            tag = eot + tag
        props += f"<{audio_tag} {' '.join(out_keys)}>{tag}</{audio_tag}>"
        tag_idx += 1
    desc = f"<music {' '.join(marks)}>{props}</music>"
    if len(offsets) == 0:
        return None
    max_offset = max(offsets)

    # dur = format_duration(max_offset - onset)
    dur = max_offset - onset
    dur = f'{dur:.1f}'
    n_tokens_st = int(onset * fps)
    n_tokens_ed = min(max_n_tokens, int(max_offset * fps)) if max_n_tokens is not None else int(max_offset * fps)
    # print('Start, End, dur', n_tokens_st, n_tokens_ed, dur)
    feature = "".join([feature_token] * (n_tokens_ed - n_tokens_st))
    head = f"<audio duration feature><duration>{dur}</duration><feature>{feature}</feature></audio>"
    return head + desc + eos, n_tokens_st, n_tokens_ed, onset, max_offset


def song2segs(song, max_sec, feature_token, eot, eos, max_n_tokens, fps, rng, drop_out, overlapping_ratio):
    dur = 0.
    segs = []
    for seg in song:
        temp = {k: seg[k] for k in seg}
        offset = float(temp["offset"])
        while offset >= dur + max_sec:
            dt = max_sec + dur - temp["onset"]
            low = int(dt * overlapping_ratio / 2)
            up = int(dt * overlapping_ratio)
            sample_sec = rng.randint(low, up) if up > low else dt
            temp["offset"] = max_sec + dur
            segs.append(temp)
            assert temp["offset"] > temp["onset"]
            yield segs2caption(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)
            temp["onset"] = sample_sec + temp["onset"]
            temp["offset"] = offset
            dur = temp["onset"]
            assert dur < offset
            segs = []
        segs.append(temp)

    if len(segs) > 0:
        assert segs[-1]["offset"] > segs[-1]["onset"]
        yield segs2caption(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)

def song2segs_revise(song, max_sec, feature_token, eot, eos, max_n_tokens, fps, rng, drop_out, overlapping_ratio, with_comparison=False):
    dur = 0.
    segs = []
    dur = max(0.0, float(song[0]["onset"]))
    for seg in song:
        temp = {k: seg[k] for k in seg}

        offset = float(temp["offset"])
        while offset >= dur + max_sec:
            dt = max_sec + dur - temp["onset"]
            low = int(dt * overlapping_ratio / 2)
            up = int(dt * overlapping_ratio)

            sample_sec = rng.randint(low, up) if up > low else dt

            temp["offset"] = max_sec + dur
            segs.append(temp)

            assert temp["offset"] > temp["onset"]
            # segs2caption_revise -> segs2caption_revise_before
            yield segs2caption_revise_before(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)
            temp["onset"] = sample_sec + temp["onset"]
            temp["offset"] = offset
            dur = temp["onset"]
            assert dur < offset
            segs = []
        segs.append(temp)

    if len(segs) > 0:
        assert segs[-1]["offset"] > segs[-1]["onset"]
        yield segs2caption_revise_before(segs, dur, feature_token, rng, drop_out, eot, eos, max_n_tokens, fps)


def create_caption(root_folder, output_folder, training_data=None, split="train", rng=None,
                   eos="<|end_of_text|>", eot="<|eot_id|>", feature_token="<|x|>",
                   max_sec=18, drop_out=False, overlapping_ratio=1,
                   save_dict=True, fps=75, selected_keys=None, with_comparison=False, add_description=False, html_first_rate=1.0, rearrange=False, grounding_param=None):
    
    # using song2segs_revise()
    if training_data is None:
        dataset_path = os.path.join(root_folder, split + ".json")
        with open(dataset_path, "r") as f:
            data = json.load(f)
    else:
        data = training_data
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
    key_mapping = {
        "tempo mean": "tempo",
        "tempo": "tempo",
        "key mode": "key",
        "onset": "onset",
        "offset": "offset",
        "mark": "mark",
        "instruments": "instruments",
        "beats": "beats",
        # "predominant instruments": "predominant-instruments",
        "predominant instruments": "instruments", 
        "instrumentation": "instruments",
        "chord progression": "chord",
        "chords": "chord",
        "chords": "chord",
        "genre": "genre",
        "genres": "genre",
        "monophonic ?": "is-monophonic",
        "time signature": "time-signature",
        "loop": "loop",
        "is_loop": "loop",
        "tempo std": "tempo-std",
        "swing ?": "is-swing",
        "swing ratio median": "swing-ratio-median",
        "swing ratio iqr": "swing-ratio-iqr",
        "ternary ?": "is-ternary",
        "vocal part": "vocal-part",
        "vocal gender": "vocal-gender",
        "emotion": "emotion",
        "melodiousness": "melodiousness",
        "articulation": "articulation",
        "rhythmic stability": "rhythmic-stability",
        "rhythmic complexity": "rhythmic-complexity",
        "dissonance": "dissonance",
        "tonal stability": "tonal-stability",
        "modality": "modality",
    }

    results = []
    for d in data:
        song = []
        for seg in d["segments"]:
            onset = seg["onset"]
            offset = seg["offset"]

            contents = {
                # "timestamp": [onset, offset]
            }
            basic_keys = ['onset', 'offset', 'mark']
            if not selected_keys:
                selected_keys = ['instruments', 'chord', 'key', 'tempo', 'beats', 'genre']
            for key in seg:

                formatted_key = key_mapping[key]
                

                if formatted_key not in basic_keys and formatted_key not in selected_keys and formatted_key not in ['rhythmic stability', 'rhythmic complexity', 'tonal stability']:
                    continue
                val = format_props(formatted_key, seg[key])
                contents[formatted_key] = val

            for selected_key in selected_keys:
                if selected_key in contents:
                    song.append(contents)
                    break
            
            # song.append(contents)

        for crop_song in song2segs_revise(song, max_sec, feature_token, eot, eos,
                                None, fps, rng, drop_out=drop_out,
                                overlapping_ratio=overlapping_ratio, with_comparison=with_comparison):
            if crop_song is None:
                continue
            desc, n_tokens_st, n_tokens_ed, onset, max_offset = crop_song

            dur = max_offset - onset
            dur = f'{dur:.1f}'
            results.append({
                "filename": d["filename"],
                "dataset": d["dataset"],
                "n_tokens_st": n_tokens_st,
                "n_tokens_ed": n_tokens_ed,
                "onset": onset,
                "offset": max_offset,
                # "duration": format_duration(max_offset - onset),
                "duration": dur,
                "caption": desc
            })

    if rearrange == 1:
        print('rearranging... 1')
        results = rearrange_single_data_0212(results)

    if rearrange == 2:
        print('rearranging... 2')
        music_first_rate = grounding_param['music_first_rate']
        grounding_only_for_change = grounding_param['grounding_only_for_change']
        
        results = rearrange_single_data_0325(results, music_first_rate, grounding_only_for_change)
    if add_description:
        print('adding description...')
        results = rearrange_0320(results, html_first_rate)

    if save_dict:
        with open(os.path.join(output_folder, f"caption_{split}.json"), "w") as f:
            json.dump(results, f, indent=2)
    return results

def rearrange_single_data(data):

    new_data = []
    tmp_dic = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G'}
    for i in range(len(data)):
        d = data[i]
        caption = d['caption']
        attribute_dic = {}
        segments_part = caption.split('</audio><music ')[-1].split('>')[0]
        segments_num = len(segments_part.split(' '))
        # print(caption.split('</audio><music ')[-1])
        
        overall_attributes = set()

        for j in range(segments_num):
            attribute_dic[tmp_dic[j]] = {}
            one_segment_part = re.findall(rf'<{tmp_dic[j]} (.*?)</{tmp_dic[j]}>', caption)[0]
            one_segment_attributes = one_segment_part.split('>')[0].split(' ')
            # timestamp 和其它
            for attribute in one_segment_attributes:
                if attribute == 'timestamp':
                    timestamp = re.findall(rf'<{attribute}>(.*?)</{attribute}>', one_segment_part)[0]
                    start = timestamp.split('-')[0]
                    start = float(start.split(':')[-1])
                    end = timestamp.split('-')[1]
                    end = float(end.split(':')[-1])
                    attribute_dic[tmp_dic[j]]['onset'] = start
                    attribute_dic[tmp_dic[j]]['offset'] = end
                else:
                    overall_attributes.add(attribute)
                    attribute_dic[tmp_dic[j]][attribute] = re.findall(rf'<{attribute}>(.*?)</{attribute}>', one_segment_part)[0]

        overall_dic = {}
        for attribute in overall_attributes:
            if attribute in ['chord', 'beats']:
                overall_dic[attribute] = ''
            elif attribute in ['key', 'tempo']:
                overall_dic[attribute] = []
            elif attribute in ['instruments']:
                overall_dic[attribute] = set()
            for j in range(segments_num):
                if attribute not in attribute_dic[tmp_dic[j]]:
                    continue
                else:
                    if attribute in ['chord', 'beats']:
                        if overall_dic[attribute] == '':
                            overall_dic[attribute] = attribute_dic[tmp_dic[j]][attribute]
                        else:
                            overall_dic[attribute] += ', ' + attribute_dic[tmp_dic[j]][attribute]
                    elif attribute in ['key', 'tempo']:
                        overall_dic[attribute].append((attribute_dic[tmp_dic[j]]['onset'], attribute_dic[tmp_dic[j]][attribute]))
                    elif attribute in ['instruments']:
                        ins_list = attribute_dic[tmp_dic[j]][attribute].split(', ')
                        for ins in ins_list:
                            overall_dic[attribute].add(ins)
            if attribute in ['tempo', 'key']:
                edit = overall_dic[attribute]
                overall_dic[attribute] = [edit[0]]
                for j in range(1, len(edit)):
                    if edit[j][1] == edit[j-1][1]:
                        continue
                    else:
                        overall_dic[attribute].append(edit[j])

        keep = caption.split('</audio>')[0] + '</audio>'
        edit = caption.split('</audio>')[-1]
        edit = edit.split('</A>')[0] + '</A>'

        start = '00:00'
        end = attribute_dic[tmp_dic[segments_num-1]]['offset']
        end = format_duration(end)

        # <music A(00:00-00:08) B(00:08-00:22)>
        ori_first_part = edit.split('>')[0] + '>'
        tgt_first_part = f"<music A({start}-{end})>"
        edit = edit.replace(ori_first_part, tgt_first_part)

        # <timestamp>00:00-00:11</timestamp>
        ori_second_part = edit.split('<timestamp>')[1].split('</timestamp>')[0]
        ori_second_part = '<timestamp>' + ori_second_part + '</timestamp>'
        tgt_second_part = ''
        edit = edit.replace(ori_second_part, tgt_second_part)
        edit = edit.replace('timestamp ', '')

        for attribute in overall_attributes:
            ori = re.findall(rf'<{attribute}>(.*?)</{attribute}>', edit)
            tgt = overall_dic[attribute]
            if attribute in ['chord', 'beats']:
                tgt = tgt
            elif attribute in ['key', 'tempo']:
                tgt = str(tgt).replace("'", "")[1:-1]
            elif attribute in ['instruments']:
                tgt = ', '.join(tgt)
            edit = edit.replace(ori[0], tgt)
        
        # print('edit:', edit)
            

        d['caption'] = keep + edit + '</music>' + '<|end_of_text|>'
        new_data.append(d)
    return new_data

def get_case_prompt(s):
    # music_part = s.split('<music A B>')[-1]
    comparison_attributes_markers = re.findall(r'<comparison \((.*?)\)>', s)[0].split(' ')
    concept_map_source = re.findall(r'<concept (.*?)>', s)[0].split(' ')
    concept_map = {}
    for marker in concept_map_source:
        term = re.findall(rf'<{marker}(.*?)</{marker}>', s)[0]
        term = re.findall(rf'<term>(.*?)</term>', term)[0]
        concept_map[marker] = term

    comparison_attributes = concept_map.values() # tempo, blabla
    # print(comparison_attributes)
    a_attributes = re.findall(r'<A (.*?)>', s)[0].split(' ')
    b_attributes = re.findall(r'<B (.*?)>', s)[0].split(' ')
    a_part = re.findall(r'<A (.*?)</A>', s)[0]
    b_part = re.findall(r'<B (.*?)</B>', s)[0]
    dic_A = {}
    
    for attribute in a_attributes:

        if attribute not in concept_map.keys():
            continue


        value = re.findall(rf'<{attribute}>(.*?)</{attribute}>', a_part)[0]
        dic_A[concept_map[attribute]] = value

    
    dic_B = {}
    for attribute in b_attributes:
        if attribute not in concept_map.keys():
            continue
        value = re.findall(rf'<{attribute}>(.*?)</{attribute}>', b_part)[0]
        dic_B[concept_map[attribute]] = value

    prompt = f"A: {str(dic_A)}\nB: {str(dic_B)}"
    return prompt

def find_first_letter_position(text):
    match = re.search(r'[a-zA-Z]', text)
    if match:
        return match.start()
    return -1


def get_chord_list(music_part):
    chord_pattern = re.compile(r'<chord>(.*?)</chord>', re.S)
    chords = chord_pattern.findall(music_part)
    res = []
    for chord_list_str in chords:
        if len(chord_list_str) == 0:
            continue
        tups = chord_list_str.split('), ')
        for tup in tups:
            if len(tup) == 0:
                continue
            if tup[-1] != ')':
                tup += ')'
            res.append(tup)

    print(res)
    assert(0)
    return chords

def rearrange_0320(results, description_first_rate=0.0):
    # split music part
    # find the caption file according to filename
    # if chord in html, use template to add a sentence describing chords
    # concatenate the music part and the description part

    query_path = 'preprocess/query_description2.json'
    with open(query_path, 'r') as f:
        query_dataset = json.load(f)
    for datapoint in results:
        filename = datapoint['filename']
        original_caption = datapoint['caption']
        audio_part = original_caption.split('</audio>')[0] + '</audio>'
        music_part = original_caption.split('</audio>')[1].split('<|end_of_text|>')[0]
        music_part1 = music_part.split('<|eot_id|>')[0]
        music_part2 = music_part.split('<|eot_id|>')[1]
        print(music_part)
        get_chord_list(music_part)
        assert(0)

        if 'AAM' in filename:
            # description = get_description_from_template()
            flag = False
        elif 'MTG' in filename:
            raw_description = query_dataset['MTG'][filename]
            s = raw_description
            if len(s) > 0 and s[0] == '\n':
                s = s[1:].strip('\n').strip(' ')
                if len(s) == 0:
                    # description = get_description_from_template()
                    flag = False
                else:
                    placeholder = 0
            else:
                chord_list = get_chord_list(music_part)


            # other use

    pass


def rearrange_single_data_0212(data, music_first_rate=1.0):

    def merge_continuous_segments(changes):
        """Merge consecutive segments with the same value (key or tempo)."""
        if not changes:
            return ""

        merged = []
        prev_start, prev_end, prev_value = changes[0]

        for start, end, value in changes[1:]:
            if value == prev_value:  # Merge if the value remains the same
                prev_end = end
            else:
                merged.append((prev_start, prev_end, prev_value))
                prev_start, prev_end, prev_value = start, end, value

        merged.append((prev_start, prev_end, prev_value))  # Append the last segment
        return " - ".join([f"{value} ({start:.1f}, {end:.1f})" for start, end, value in merged])

    def parse_music_part(music_part):
        """Parse and merge multiple sections in a music_part string."""
        sections = re.findall(r'<([A-Z]) .*?>(.*?)</\1>', music_part, re.DOTALL)

        timestamps = []
        tempo_changes = []
        key_changes = []
        instruments_set = set()
        chords = []
        beatss = []
        first_order = []  

        for i, (section_name, section_content) in enumerate(sections):
            timestamp_match = re.search(r'<timestamp>\(([\d.]+), ([\d.]+)\)</timestamp>', section_content)
            if timestamp_match:
                start_time, end_time = float(timestamp_match.group(1)), float(timestamp_match.group(2))
                timestamps.append((start_time, end_time))
            else:
                continue

            attributes = {}
            for match in re.finditer(r'<(tempo|key|instruments|chord|beats)>(.*?)</\1>', section_content, re.DOTALL):
                attr_name, attr_value = match.groups()
                attributes[attr_name] = attr_value.strip()
                if i == 0 and attr_name not in first_order:
                    first_order.append(attr_name) 
            
            if "tempo" in attributes:
                tempo_changes.append((start_time, end_time, attributes["tempo"]))

            if "key" in attributes:
                key_changes.append((start_time, end_time, attributes["key"]))

            if "instruments" in attributes and attributes["instruments"]:
                instruments_set.update(attributes["instruments"].split(', '))

            if "chord" in attributes:
                chord_match = re.findall(r'\(([\d.]+),\s*([^)]+)\)', attributes["chord"])
                for chord_time, chord_name in chord_match:
                    chords.append((float(chord_time), chord_name.strip()))

            if "beats" in attributes:
                beats_match = re.findall(r'\(([\d.]+),\s*([^)]+)\)', attributes["beats"])
                for beats_time, beats_name in beats_match:
                    beatss.append((float(beats_time), beats_name.strip()))

        if not timestamps:
            return None

        global_start, global_end = min(t[0] for t in timestamps), max(t[1] for t in timestamps)

        key_str = key_changes[0][2] if len(set(k[2] for k in key_changes)) == 1 else merge_continuous_segments(key_changes)
        tempo_str = tempo_changes[0][2] if len(set(t[2] for t in tempo_changes)) == 1 else merge_continuous_segments(tempo_changes)
        if '-' not in key_str:
            key_str = f"{key_str} ({global_start:.1f}, {global_end:.1f})"
        if '-' not in tempo_str:
            tempo_str = f"{tempo_str} ({global_start:.1f}, {global_end:.1f})"

        chords.sort()
        chord_str = ", ".join([f"({time:.2f}, {name})" for time, name in chords])
        beatss.sort()
        beats_str = ", ".join([f"({time:.2f}, {name})" for time, name in beatss])
        instruments_str = ", ".join(sorted(instruments_set)) if instruments_set else ""

        attr_dict = {
            "timestamp": f"({global_start:.2f}, {global_end:.2f})",
            "tempo": tempo_str,
            "key": key_str,
            "instruments": instruments_str,
            "chord": chord_str,
            "beats": beats_str
        }

        attr_list = " ".join(first_order)  
        # merged_music = f"<music timestamp {attr_list}>\n  <timestamp>{attr_dict['timestamp']}</timestamp><|eot_id|>"
        merged_music = f"<music timestamp {attr_list}><timestamp>{attr_dict['timestamp']}</timestamp><|eot_id|>"
        for attr in first_order:
            # merged_music += f"\n  <{attr}>{attr_dict[attr]}</{attr}>"
            merged_music += f"<{attr}>{attr_dict[attr]}</{attr}>"
        # merged_music += "\n</music><|end_of_text|>"
        merged_music += "</music><|end_of_text|>"

        return merged_music

    new_data = []
    for d in data:
        cap = d['caption']
        music_part = cap.split('</audio>')[-1]
        parsed_music_part = parse_music_part(music_part)
        d['caption'] = cap.split('</audio>')[0] + '</audio>' + parsed_music_part
        new_data.append(d)

    return new_data


def rearrange_single_data_0325(results, music_first_rate=1.0, grounding_only_for_change=False):
    import random
    new_data = []
    original_data = results 
    def merge_continuous_segments(changes):
        """Merge consecutive segments with the same value (key or tempo)."""
        if not changes:
            return ""

        merged = []
        prev_start, prev_end, prev_value = changes[0]

        for start, end, value in changes[1:]:
            if value == prev_value:  # Merge if the value remains the same
                prev_end = end
            else:
                merged.append((prev_start, prev_end, prev_value))
                prev_start, prev_end, prev_value = start, end, value

        merged.append((prev_start, prev_end, prev_value))  # Append the last segment
        return " - ".join([f"{value} ({start:.1f}, {end:.1f})" for start, end, value in merged])

    for index in range(len(original_data)):
        caption = original_data[index]['caption']
        audio_part = caption.split('</audio>')[0] + '</audio>'
        info = caption.split('</audio>')[1]
        music_part_1 = info.split('<|eot_id|>')[0] # <|eot_id|>
        music_part_2 = info.split('<|eot_id|>')[-1].split('<|end_of_text|>')[0] # <|end_of_text|>

        sections = re.findall(r'<([A-Z]) .*?>(.*?)</\1>', info, re.DOTALL)

        timestamps = []
        tempo_changes = []
        key_changes = []
        instruments_set = set()
        chords = []
        beatss = []
        first_order = []  

        for i, (section_name, section_content) in enumerate(sections):
            timestamp_match = re.search(r'<timestamp>\(([\d.]+), ([\d.]+)\)</timestamp>', section_content)
            if timestamp_match:
                start_time, end_time = float(timestamp_match.group(1)), float(timestamp_match.group(2))
                timestamps.append((start_time, end_time))
            else:
                continue

            attributes = {}
            for match in re.finditer(r'<(tempo|key|instruments|chord|beats)>(.*?)</\1>', section_content, re.DOTALL):
                attr_name, attr_value = match.groups()
                attributes[attr_name] = attr_value.strip()
                if i == 0 and attr_name not in first_order:
                    first_order.append(attr_name) 
            
            if "tempo" in attributes:
                tempo_changes.append((start_time, end_time, attributes["tempo"]))

            if "key" in attributes:
                key_changes.append((start_time, end_time, attributes["key"]))

            if "instruments" in attributes and attributes["instruments"]:
                instruments_set.update(attributes["instruments"].split(', '))

            if "chord" in attributes:
                chord_match = re.findall(r'\(([\d.]+),\s*([^)]+)\)', attributes["chord"])
                for chord_time, chord_name in chord_match:
                    chords.append((float(chord_time), chord_name.strip()))

            if "beats" in attributes:
                beats_match = re.findall(r'\(([\d.]+),\s*([^)]+)\)', attributes["beats"])
                for beats_time, beats_name in beats_match:
                    beatss.append((float(beats_time), beats_name.strip()))

        key_str = key_changes[0][2] if len(set(k[2] for k in key_changes)) == 1 else merge_continuous_segments(key_changes)
        tempo_str = tempo_changes[0][2] if len(set(t[2] for t in tempo_changes)) == 1 else merge_continuous_segments(tempo_changes)
        global_start, global_end = min(t[0] for t in timestamps), max(t[1] for t in timestamps)



        if grounding_only_for_change and '-' not in key_str and '-' not in tempo_str:
            new_caption = f'{audio_part}<analysis music>{music_part_1}<|eot_id|>{music_part_2}</analysis><|end_of_text|>'
            original_data[index]['caption'] = new_caption
            new_data.append(original_data[index])
            continue
        # create grounding part
        qa_dic = {}
        if '-' in tempo_str:
            for triple in tempo_str.split(' - '):
                value = triple.split(' (')[0]
                start_and_end = '(' + triple.split(' (')[1]
                qa_dic[value] = start_and_end
        else:
            qa_dic[tempo_str] = f'({global_start:.1f}, {global_end:.1f})'

        if '-' in key_str:
            for triple in key_str.split(' - '):
                value = triple.split(' (')[0]
                start_and_end = '(' + triple.split(' (')[1]
                qa_dic[value] = start_and_end
        else:
            qa_dic[key_str] = f'({global_start:.1f}, {global_end:.1f})'

        # change the qa_dic to list and shuffle it
        qa_list = list(qa_dic.items())
        random.shuffle(qa_list)
        
        query_part = ""
        for j in range(1, len(qa_list)+1):
            query_part += f'<Q{j}>{qa_list[j-1][0]}</Q{j}>'
        tmp_q = ' '.join(f'Q{j}' for j in range(1, len(qa_list)+1))
        query_part = f"<query {tmp_q}>{query_part}</query>"

        answer_part = ""
        for j in range(1, len(qa_list)+1):
            answer_part += f'<A{j}>{qa_list[j-1][1]}</A{j}>'
        tmp_a = ' '.join(f'A{j}' for j in range(1, len(qa_list)+1))
        # whole_answer_part = f"<answer {tmp_a}>{answer_part}</answer>"
        grounding_part_behind = f"<grounding query answer>{query_part}<answer {tmp_a}>{answer_part}</answer></grounding>"
        grounding_part_front = f"<grounding query answer>{query_part}<answer {tmp_a}><|eot_id|>{answer_part}</answer></grounding>"
        

        if random.random() < music_first_rate:
            new_caption = f'{audio_part}<analysis music grounding>{music_part_1}<|eot_id|>{music_part_2}{grounding_part_behind}</analysis><|end_of_text|>'
        else:
            new_caption = f'{audio_part}<analysis grounding music>{grounding_part_front}{music_part_1}{music_part_2}</analysis><|end_of_text|>'
        

        original_data[index]['caption'] = new_caption
        new_data.append(original_data[index])

    return new_data