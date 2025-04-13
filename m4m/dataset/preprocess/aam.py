import os
import json
import librosa


def read_chords(path):
    with open(path, "r") as f:
        lines = f.readlines()

    chords = []
    for line in lines:
        line = line.rstrip()
        if str.startswith(line, "@") or line == "":
            continue
        chords.append(line.split(",")[-1][1:-1])
    return " - ".join(remove_redundant(chords))


def remove_redundant(data, shrink=False):
    pre = 0
    res = []
    for ch in data:
        if not pre == ch:
            res.append(ch)
        pre = ch
    if shrink:
        if len(res) == 1:
            return res[0]
    return res


def split_line(line):
    line = line.split("','")
    x = [b for a in line[0].split("',") for b in a.split(",'")]
    return x + [line[1][1:-1].split(","), line[2][1:-2].split(",")]

def merge_elements(elements):
    new_elements = []
    cur_element = elements[0][1]
    cur_time = elements[0][0]
    for element_tuple in elements:
        time = element_tuple[0]
        element = element_tuple[1]
        if element == cur_element:
            continue
        else:
            new_elements.append((cur_time, cur_element))
            cur_element = element
            cur_time = time
    
    new_elements.append((cur_time, cur_element))
    return new_elements


def read_segment_chords(segment_onset, segment_offset, chord_path):
    chords = []
    with open(chord_path, "r") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        if str.startswith(line, "@") or line == "":
            continue
        if i == len(lines) - 1:
            break
        onset = line.split(',')[0]
        offset = lines[i+1].split(',')[0]
        if float(onset) < float(segment_onset):
            continue
        if float(onset) >= float(segment_offset):
            break
        chord = line.split(',')[3][1:-1]
        chords.append((onset, chord))

    chords = merge_elements(chords)
    round_chords = []
    for chord in chords:
        time = chord[0]
        time = round(float(time), 2)
        round_chords.append((time, chord[1]))
    return round_chords

def read_segment_beats(segment_onset, segment_offset, beat_path):
    beats = []
    with open(beat_path, "r") as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        line = line.rstrip()
        if str.startswith(line, "@") or line == "":
            continue
        if i == len(lines) - 1:
            break
        onset = line.split(',')[0]
        offset = lines[i+1].split(',')[0]
        if float(onset) < float(segment_onset):
            continue
        if float(onset) >= float(segment_offset):
            break
        beat_mark = 'downbeat' if line.split(',')[2] == '1' else 'beat'
        beats.append((onset, beat_mark))

    round_beats = []
    for beat in beats:
        time = beat[0]
        time = round(float(time), 2)
        round_beats.append((str(time), beat[1]))

    first_downbeat_idx = -1
    period = -1
    for i in range(len(round_beats)):
        tup = round_beats[i]
        if tup[1] == 'downbeat':
            if first_downbeat_idx == -1:
                first_downbeat_idx = i
            elif period == -1:
                period = i - first_downbeat_idx

            round_beats[i] = (tup[0], '0')

    last_downbeat_idx = first_downbeat_idx
    
    for i in range(last_downbeat_idx, len(round_beats)):
        if round_beats[i][1] == '0':
            last_downbeat_idx = i
        else:
            round_beats[i] = (round_beats[i][0], str(i - last_downbeat_idx))

    for i in range(first_downbeat_idx):
        round_beats[i] = (round_beats[i][0], str(period - last_downbeat_idx + i))
        

    # beats = merge_elements(beats)
    # print(beats)
    return round_beats

def read_segment_instruments(segment_onset, segment_offset, onset_path):
    instruments = []
    with open(onset_path, "r") as f:
        lines = f.readlines()

    instruments_name = {}
    cnt = 1
    for i, line in enumerate(lines):
        line = line.rstrip()
        if str.startswith(line, "@"):
            if "Onset events of" in line:
                instruments_name[cnt] = line[28:].split("'")[0]
                cnt += 1
            continue
            # print(line)

        if line == "":
            continue

        if i == len(lines) - 1:
            break

        onset = line.split(',')[0]
        offset = lines[i+1].split(',')[0]
        if float(onset) < float(segment_onset):
            continue    
        if float(onset) >= float(segment_offset):
            break

        linedata = line.split('[')
        instrument_list = []

        for i in range(1, len(linedata)):
            v = linedata[i]
            if len(v) > 4:
                instrument_list.append(instruments_name[i])
        instruments.append((onset, instrument_list))
                
    instruments = merge_elements(instruments)
    return instruments


def read_segs(path):
    print(path)
    with open(path, "r") as f:
        lines = f.readlines()

    segments = []
    for i, line in enumerate(lines):
        line = line.rstrip()
        if str.startswith(line, "@") or line == "":
            continue
        line = split_line(line)
        if i == len(lines) - 1:
            break
        
        onset = float(line[0])
        offset = float(lines[i + 1].split(",")[0])
        chord_path = str.replace(path, "segments", "beatinfo")
        onset_path = str.replace(path, "segments", "onsets")
        chords = read_segment_chords(onset, offset, chord_path)
        detailed_instruments = read_segment_instruments(onset, offset, onset_path)
        beats = read_segment_beats(onset, offset, chord_path)
        segments.append({
            "onset": onset,
            "offset": offset,
            "mark": line[1],
            "tempo mean": line[2],
            "key mode": line[3],
            "instruments": line[4],
            "chord progression": chords,
            "beats": beats,
            # "instruments": detailed_instruments,
        })

    return segments, segments[-1]["offset"]


def process(root_folder, output_folder):
    annotations = os.path.join(root_folder, "annotations")
    audio_folder = os.path.join(root_folder, "mix")
    res = []
    for song in os.listdir(audio_folder):
        path = os.path.join(audio_folder, song)
        # chords_path = os.path.join(annotations, str.replace(song, "_mix.flac", "_beatinfo.arff"))
        segs_path = os.path.join(annotations, str.replace(song, "_mix.flac", "_segments.arff"))
        # chords = read_chords(chords_path)
        segments, duration = read_segs(segs_path)
        res.append({
            "filename": path,
            # "duration": duration,
            "segments": segments,
        })
        print(res[-1])

    with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
        json.dump(res, jsonfile, indent=2)

    keys = {}
    for data in res:
        for key in data:
            keys[key] = 0
    with open(os.path.join(output_folder, "keys.lst"), "w") as f:
        f.write("\n".join([k for k in keys]))
