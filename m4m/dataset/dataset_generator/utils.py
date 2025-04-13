import re
import random
import itertools
import numpy as np
# import openai

def remove_parentheses_content(chord, replace='*'):
    # * 代表有括号存在过
    return re.sub(r'\([^)]*\)', replace, chord)

def exclude_keys(keys, excluded):
    res = []
    for k in keys:
        if not k in excluded:
            res.append(k)

    return res


def format_sample(sample):
    res = []
    for s in sample:
        d = {}
        for k in s:
            v = s[k]
            if k == "duration":
                v = float(s[k])
            elif k == "tempo mean":
                if s[k] in ["None", "none", ""]:
                    continue
                v = str(float(s[k]))
            elif k == "tempo":
                v = str.replace(s[k], " - ", " bpm - ") + " bpm"
            elif k == "key mode":
                v = str.replace(s[k], ": ", "")
                v = str.replace(v, ":", "")
            d[k] = v
        res.append(d)
    return res


dic_root = {
    'Ab': 'G#',
    'Bb': 'A#',
    'Cb': 'B',
    'Db': 'C#',
    'Eb': 'D#',
    'Fb': 'E',
    'Gb': 'F#',
}


def format_props(key, val):
    if key == "tempo":
        return int(float(val))
    if key == "key":
        replace_dict = {":": "", "major": "maj", "minor": "min"}
        for name in replace_dict:
            val = str.replace(val, name, replace_dict[name])
        val = str.replace(val, "maj", " major")
        val = str.replace(val, "min", " minor")
        val = str.replace(val, " ", "")
        val = val.split("major")
        scale, mode = [val[0].split("minor")[0], "minor"] if len(val) == 1 else [val[0], "major"]
        # convert all b to #
        if 'b' in scale:
            scale = dic_root[scale]
        return f"{scale}{mode}"
    if key == "chord":
        ls = []
        if type(val) is list:
            for tup in val:
                replace_dict = {":": "", "major": "maj", "minor": "min"}
                for name in replace_dict:
                    tup[1] = str.replace(tup[1], name, replace_dict[name])
                tup[1] = str.replace(tup[1], "maj", "major")
                tup[1] = str.replace(tup[1], "min", "minor")
                ls.append([float(tup[0]), tup[1]])
            # return str(ls)[1:-1]
        # print(ls)
        return ls
    
    if key == "beats":
        ls = []
        if type(val) is list:
            for tup in val:
                if int(tup[1]) < 0:
                    continue
                ls.append([float(tup[0]),tup[1]])
        return ls

    if key == "instruments":
        if type(val) is dict:
            val = [k for k in val if val[k]]
        return val

    if key == "genre":
        valid_genres = [
            'hip hop',
            'hiphop',
            'electronic',
            'cinematic',
            'pop',
            'jazz',
            'Jazz',
            'blues',
            'classical',
            'country',
            'disco',
            'metal',
            'reggae',
            'rock',
            'Rock'
        ]
        if type(val) == type('str'):
            val = [val]
        return str(val)
        # if type(val) == type([]):
        #     new_str = ""
        #     new_val = []
        #     for i in range(len(val)):
        #         if val[i] not in valid_genres:
        #             continue
        #         elif val[i] == 'hip hop':
        #             new_val.append('hiphop')
        #         elif val[i] == 'Jazz':
        #             new_val.append('jazz')
        #         elif val[i] == 'Rock':
        #             new_val.append('rock')
        #         else:
        #             new_val.append(val[i])

        #     for i in range(len(new_val)):
        #         if i != 0:
        #             new_str += ", "
        #         new_str += new_val[i]
        #     return new_str if len(new_str) > 0 else None
        # else:
        #     if val not in valid_genres:
        #         return val
        #     elif val == 'hip hop':
        #         val = 'hiphop'
        #     elif val == 'Jazz':
        #         val = 'jazz'
        #     elif val == 'Rock':
        #         val = 'rock'
        #     else:
        #         return val

    if "is_" in key:
        if type(val) is bool:
            return str(val).lower()
        if "true" in val or "yes" in val or "True" in val:
            return "true"
        elif "false" in val or "no" in val or "False" in val:
            return "false"
        else:
            return "unkown"
        
    if key in ['melodiousness', 'articulation', 'rhythmic-stability', 'rhythmic-complexity', 'dissonance', 'tonal-stability', 'modality']:
        val = round(float(val))

    return val




def format_props2(key, val):
    if key == "tempo":
        return int(float(val))
    if key == "key":
        replace_dict = {":": "", "major": "maj", "minor": "min"}
        for name in replace_dict:
            val = str.replace(val, name, replace_dict[name])
        val = str.replace(val, "maj", " major")
        val = str.replace(val, "min", " minor")
        val = str.replace(val, " ", "")
        val = val.split("major")
        scale, mode = [val[0].split("minor")[0], "minor"] if len(val) == 1 else [val[0], "major"]

        return f"{scale} {mode}"
    if key == "chord":
        ls = []
        if type(val) is list:
            for tup in val:
                replace_dict = {":": "", "major": "maj", "minor": "min"}
                for name in replace_dict:
                    tup[1] = str.replace(tup[1], name, replace_dict[name])
                tup[1] = str.replace(tup[1], "maj", "major")
                tup[1] = str.replace(tup[1], "min", "minor")
                ls.append([float(tup[0]), tup[1]])
            # return str(ls)[1:-1]
        return ls

    if key == "instruments":
        if type(val) is dict:
            val = [k for k in val if val[k]]
            if 'chords' in val:
                val.remove('chords')
            if 'melody' in val:
                val.remove('melody')
        return val

    if key == "genre":
        valid_genres = [
            'hip hop',
            'hiphop',
            'electronic',
            'cinematic',
            'pop',
            'jazz',
            'Jazz',
            'blues',
            'classical',
            'country',
            'disco',
            'metal',
            'reggae',
            'rock',
            'Rock'
        ]
        if type(val) == type([]):
            new_str = ""
            new_val = []
            for i in range(len(val)):
                if val[i] not in valid_genres:
                    continue
                elif val[i] == 'hip hop':
                    new_val.append('hiphop')
                elif val[i] == 'Jazz':
                    new_val.append('jazz')
                elif val[i] == 'Rock':
                    new_val.append('rock')
                else:
                    new_val.append(val[i])

            for i in range(len(new_val)):
                if i != 0:
                    new_str += ", "
                new_str += new_val[i]
            return new_str if len(new_str) > 0 else None
        else:
            if val not in valid_genres:
                return None
            elif val == 'hip hop':
                val = 'hiphop'
            elif val == 'Jazz':
                val = 'jazz'
            elif val == 'Rock':
                val = 'rock'
            else:
                return val

    if "is_" in key:
        if "true" in val or "yes" in val or "True" in val:
            return "true"
        elif "false" in val or "no" in val or "False" in val:
            return "false"
        else:
            return "unkown"

    return val


def format_props3(key, val):
    if key == "tempo":
        return int(float(val))
    if key == "key":
        replace_dict = {":": "", "major": "maj", "minor": "min"}
        for name in replace_dict:
            val = str.replace(val, name, replace_dict[name])
        val = str.replace(val, "maj", " major")
        val = str.replace(val, "min", " minor")
        val = str.replace(val, " ", "")
        val = val.split("major")
        scale, mode = [val[0].split("minor")[0], "minor"] if len(val) == 1 else [val[0], "major"]

        return f"{scale} {mode}"
    if key == "chord":
        ls = []
        if type(val) is list:
            for tup in val:
                replace_dict = {":": "", "major": "maj", "minor": "min"}
                for name in replace_dict:
                    tup[1] = str.replace(tup[1], name, replace_dict[name])
                tup[1] = str.replace(tup[1], "maj", "major")
                tup[1] = str.replace(tup[1], "min", "minor")
                # 去掉转位标记
                tup[1] = tup[1].split("/")[0]
                # 去掉括号，有括号的标记
                if '(' in tup[1] and ')' in tup[1]:
                    tup[1] = remove_parentheses_content(tup[1])
                ls.append([float(tup[0]), tup[1]])
            # return str(ls)[1:-1]
        return ls

    if key == "instruments":
        if type(val) is dict:
            val = [k for k in val if val[k]]
            if 'chords' in val:
                val.remove('chords')
            if 'melody' in val:
                val.remove('melody')
        return val

    if key == "genre":
        valid_genres = [
            'hip hop',
            'hiphop',
            'electronic',
            'cinematic',
            'pop',
            'jazz',
            'Jazz',
            'blues',
            'classical',
            'country',
            'disco',
            'metal',
            'reggae',
            'rock',
            'Rock'
        ]
        if type(val) == type([]):
            new_str = ""
            new_val = []
            for i in range(len(val)):
                if val[i] not in valid_genres:
                    continue
                elif val[i] == 'hip hop':
                    new_val.append('hiphop')
                elif val[i] == 'Jazz':
                    new_val.append('jazz')
                elif val[i] == 'Rock':
                    new_val.append('rock')
                else:
                    new_val.append(val[i])

            for i in range(len(new_val)):
                if i != 0:
                    new_str += ", "
                new_str += new_val[i]
            return new_str if len(new_str) > 0 else None
        else:
            if val not in valid_genres:
                return None
            elif val == 'hip hop':
                val = 'hiphop'
            elif val == 'Jazz':
                val = 'jazz'
            elif val == 'Rock':
                val = 'rock'
            else:
                return val

    if "is_" in key:
        if "true" in val or "yes" in val or "True" in val:
            return "true"
        elif "false" in val or "no" in val or "False" in val:
            return "false"
        else:
            return "unkown"

    return val

def crop_props(key, val, onset, offset, seg_onset, aug=False, tempo_dt=0):
    if key == "tempo":
        val = round(val + val * tempo_dt) if aug else round(val)
        val = str(val) + " bpm"

    elif key == "chord":
        chord = [[sample[0] - seg_onset if sample[0] - seg_onset > 0 else 0, sample[1]] for i, sample in enumerate(val)
                 if
                 (i == len(val) - 1 or val[i + 1][0] > onset) and sample[0] < offset]
        # if len(chord) == 0:
        #     print(onset, offset, seg_onset)
        #     print(val)

        val = ", ".join([f"({round(sample[0], 2)}, {sample[1]})" for sample in chord])
    elif key == "instruments":
        val = ", ".join(val)

    return val


def crop_props_revise(key, val, onset, offset, seg_onset, aug=False, tempo_dt=0):
    if key == "tempo":
        val = round(val + val * tempo_dt) if aug else round(val)
        val = str(val) + " bpm"

    elif key == "chord" or key == "beats":
        chord = [[sample[0] - seg_onset if sample[0] - seg_onset > 0 else 0, sample[1]] for i, sample in enumerate(val)
                 if
                 (i == len(val) - 1 or val[i + 1][0] > onset) and sample[0] < offset]
        # if len(chord) == 0:
        #     print(onset, offset, seg_onset)
        #     print(val)

        val = ", ".join([f"({round(sample[0], 2)}, {sample[1]})" for sample in chord])

    elif key == "instruments":
        if type(val) == type([]):
            val = ", ".join(val)

    return val

def format_duration(t):
    t = int(t)
    sec = t % 60
    min = (t // 60) % 60
    return "{:02d}:{:02d}".format(int(min), int(sec))


def divide_into_more_segs(seg, rng):
    onset = float(seg["onset"]) * 100
    offset = float(seg["offset"]) * 100

    if offset - onset < 8 * 100:
        return [seg]
    elif offset - onset < 10 * 100:
        n = rng.randint(1, 3)
    else:
        n = rng.randint(1, 4)
    if n == 1:
        return [seg]
    seg_onsets = [float(seg["onset"])]
    ed = round(offset - n * 100 + 1, 2)
    while n > 1:

        next_onset = rng.randint(round(onset + 2 * 100, 2), ed)
        seg_onsets.append(next_onset / 100.)
        n -= 1
        onset = next_onset
        if ed - onset < 4 * 100:
            break

    seg_onsets.append(float(seg["offset"]))
    res = []
    for i, s in enumerate(seg_onsets[:-1]):
        data = {k: seg[k] for k in seg}
        data["onset"] = seg_onsets[i]
        data["offset"] = seg_onsets[i + 1]
        res.append(data)
    return res

def get_pairs(idx, rng):
    assert idx > 1
    KEEP_RATE = 1.01
    audios = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'][0:idx]
    rng.shuffle(audios)

    combinations = list(itertools.combinations(audios, 2))
    combinations = [comb for comb in combinations if rng.rand() < KEEP_RATE]
    pairs = [f'({comb[0]}, {comb[1]})' for comb in combinations]
    return pairs
    
def get_pure_comparison(key, val1, val2):
    if key == 'tempo':
        val1, val2 = int(val1), int(val2)
        if val1 > val2:
            content = 'faster'
        elif val1 == val2:
            content = 'same'
        else:
            content = 'slower'
    elif key == 'key':
        val1_root, val1_quality = val1.split(' ')[0], val1.split(' ')[1]
        val2_root, val2_quality = val2.split(' ')[0], val2.split(' ')[1]
        if val1_root == val2_root and val1_quality == val2_quality:
            content = 'same'
        elif val1_root == val2_root: # 例如C major 和 C minor
            # content = 'parallel keys'
            content = 'same keys'
        elif val1_quality == val2_quality:
            content = 'same modes'
        else:
            content = 'different'
    else:
        content = 'unknown'
    return content
        

def get_comparison(pairs, keys, comp_dic):
    audios = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    res = ''
    for pair in pairs:
        audio1 = pair[1:2]
        audio2 = pair[-2:-1]
        audio1_idx = audios.index(audio1)
        audio2_idx = audios.index(audio2)
        selected_keys = keys
        comparison = ''
        for key in selected_keys:
            val1 = comp_dic[key][audio1_idx]
            val2 = comp_dic[key][audio2_idx]
            one_comparison = f'<{key}>{get_pure_comparison(key, val1, val2)}</{key}>'
            comparison += one_comparison

        res += f"<{pair} {' '.join(selected_keys)}>{comparison}</{pair}>"
    return res


def get_natural_comparison(music_dic1, music_dic2, comparison_used_key):
    prompt_dic = {'A':{}, 'B':{}}
    for key in comparison_used_key:
        val1 = music_dic1[key]
        val2 = music_dic2[key]
        prompt_dic['A'][key] = val1
        prompt_dic['B'][key] = val2

    base_prompt = "I have a pair of music A and B, I would like you to compare the \
        two pieces of music and summarize the comparison in 80 words. You must mention \
        the music attribute I gave you in each group. Moreover, you can inference \
        more music information not limited on it based on your musical knowledge.\n"
    
    prompt = base_prompt + str(prompt_dic)
    print(prompt)

    from openai import OpenAI
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key='xxx',
    )
    print('getting response from chatgpt...')
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    print(response.choices[0].message.content.strip())
    return response.choices[0].message.content.strip()

def get_sequence_data(v, start, end):
    # when splitting pair, only keep the chords and beats between start, end
    res = []
    for tup in v:
        if float(tup[0]) >= start and float(tup[0]) < end:
            res.append(tup)
    return res

def reformat_sequence_data(data, onset, offset):
    # when creating caption, 1. the format should be （t, v）, (t, v) ... 2. the time should be aligned to start from 0
    res = []
    for tup in data:
        res.append((round(float(tup[0]) - onset, 2), tup[1]))
    return res