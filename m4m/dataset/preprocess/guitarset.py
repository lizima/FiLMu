import os
import json
import librosa

def read_chords(path):
    chords = []

    with open(path, 'r') as file:
        json_data = json.load(file)
    annotations = json_data["annotations"]
    for d in annotations:
        if d["namespace"] == "chord"  and len(d['annotation_metadata']['annotation_rules']) > 0:
            for val in d["data"]:
                chord = val["value"]
                time = val['time']
                duration = val['duration']
                chords.append((time, chord))
                # segments.append({
                #     "chord": chord,
                #     "onset": time,
                #     "duration": time+duration,
                # })
    return chords

def rearange(dic):
    new_dic = {}
    segments = [{'mark': 'M'}]
    for k, v in dic.items():
        if k == 'filename':
            new_dic[k] = v
        elif k == 'duration':
            # new_dic[k] = v
            # segments[0][k] = v
            segments[0]['onset'] = 0
            segments[0]['offset'] = v
        elif k == "title" or k == "artist" or k == "beats by measure":
            continue
        else:
            segments[0][k] = v
    new_dic['segments'] = segments
    return new_dic

def process(root_folder, output_folder):
    res = []
    for song in os.listdir(root_folder):
        data = {}
        if not str.endswith(song, ".jams"):
            continue
        path = os.path.join(root_folder, song)
        with open(path, 'r') as file:
            json_data = json.load(file)
            annotations = json_data["annotations"]
            file_metadata = json_data["file_metadata"]
            # sandbox = json_data["sandbox"]
            for d in file_metadata:
                if d in ["jams_version"]:
                    continue
                val = file_metadata[d]
                if val in [""] or (type(val) is dict and len(val) == 0):
                    continue
                if d == "title":

                    file_path = str.replace(path, ".jams", "_mix.wav")
                    data["filename"] = file_path

                else:
                    data[d] = val
            midis = []
            for d in annotations:
                if d["namespace"] == "key_mode":
                    assert len(d["data"]) == 1
                    data["key mode"] = d["data"][0]["value"]
                elif d["namespace"] == "tempo":
                    assert len(d["data"]) == 1
                    data["tempo mean"] = d["data"][0]["value"]
                elif d["namespace"] == "beat_position":
                    meter = []
                    beats = d["data"]
                    n_m = 0
                    pre_mid = 1
                    beat_by_measure = []

                    for b in beats:
                        m = str(b["value"]["num_beats"]) + "/" + str(b["value"]["beat_units"])
                        if len(meter) == 0 or not m == meter[-1]:
                            meter.append(m)
                        m_id = b["value"]["measure"]
                        if m_id == pre_mid:
                            n_m += 1
                        else:
                            pre_mid = m_id
                            beat_by_measure.append(n_m)
                            n_m = 1
                    beat_by_measure.append(n_m)
                    if len(meter) == 0:
                        continue
                    assert len(meter) == 1
                    data["time signature"] = meter[0]
                    data["beats by measure"] = beat_by_measure
                elif d["namespace"] == "chord":
                    chords = []
                    for val in d["data"]:
                        val = val["value"]
                        if len(chords) == 0 or not val == chords[-1]:
                            chords.append(val)
                    if len(chords) == 0:
                        continue
                    data["chord progression"] = " - ".join(chords)
                elif d["namespace"] == "pitch_contour":
                    pass
                elif d["namespace"] == "note_midi":
                    trk = []
                    for note in d["data"]:
                        trk.append({
                                    "time": note["time"],
                                    "duration": note["duration"],
                                    "pitch": round(float(note["value"]))})
                    if len(trk) > 0:
                        midis.append({
                            "source": d["annotation_metadata"]["data_source"],
                            "instrument": "guitar",
                            "notes": trk
                        })
            if len(midis) > 0:
                name = data["filename"].split("/")[-1]
                midi_path = os.path.join(output_folder, "midis")
                os.makedirs(midi_path, exist_ok=True)
                midi_path = os.path.join(midi_path, name + ".json")
                with open(midi_path, "w") as jsonfile:
                    json.dump(midis, jsonfile, indent=2)
                # data["midi"] = midi_path
        data["instruments"] = "guitar"
        data["monophonic ?"] = "no"

        chords = read_chords(path)
        data["chord progression"] = chords
        data = rearange(data)
        res.append(data)
        print(data)


    # pitch_contour
    # note_midi
    # beat_position
    # tempo
    # chord
    # key_mode

    with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
         json.dump(res, jsonfile, indent=2)

    keys = {}
    for data in res:
        for key in data:
            keys[key] = 0
    with open(os.path.join(output_folder, "keys.lst"), "w") as f:
        f.write("\n".join([k for k in keys]))



# import os
# import json
# import librosa


# def process(root_folder, output_folder):
#     res = []
#     for song in os.listdir(root_folder):
#         data = {}
#         if not str.endswith(song, ".jams"):
#             continue
#         path = os.path.join(root_folder, song)
#         with open(path, 'r') as file:
#             json_data = json.load(file)
#             annotations = json_data["annotations"]
#             file_metadata = json_data["file_metadata"]
#             # sandbox = json_data["sandbox"]
#             for d in file_metadata:
#                 if d in ["jams_version"]:
#                     continue
#                 val = file_metadata[d]
#                 if val in [""] or (type(val) is dict and len(val) == 0):
#                     continue
#                 if d == "title":

#                     file_path = str.replace(path, ".jams", "_mix.wav")
#                     data["filename"] = file_path

#                 else:
#                     data[d] = val
#             midis = []
#             for d in annotations:
#                 if d["namespace"] == "key_mode":
#                     assert len(d["data"]) == 1
#                     data["key mode"] = d["data"][0]["value"]
#                 elif d["namespace"] == "tempo":
#                     assert len(d["data"]) == 1
#                     data["tempo mean"] = d["data"][0]["value"]
#                 elif d["namespace"] == "beat_position":
#                     meter = []
#                     beats = d["data"]
#                     n_m = 0
#                     pre_mid = 1
#                     beat_by_measure = []

#                     for b in beats:
#                         m = str(b["value"]["num_beats"]) + "/" + str(b["value"]["beat_units"])
#                         if len(meter) == 0 or not m == meter[-1]:
#                             meter.append(m)
#                         m_id = b["value"]["measure"]
#                         if m_id == pre_mid:
#                             n_m += 1
#                         else:
#                             pre_mid = m_id
#                             beat_by_measure.append(n_m)
#                             n_m = 1
#                     beat_by_measure.append(n_m)
#                     if len(meter) == 0:
#                         continue
#                     assert len(meter) == 1
#                     data["time signature"] = meter[0]
#                     data["beats by measure"] = beat_by_measure
#                 elif d["namespace"] == "chord":
#                     chords = []
#                     for val in d["data"]:
#                         val = val["value"]
#                         if len(chords) == 0 or not val == chords[-1]:
#                             chords.append(val)
#                     if len(chords) == 0:
#                         continue
#                     data["chord progression"] = " - ".join(chords)
#                 elif d["namespace"] == "pitch_contour":
#                     pass
#                 elif d["namespace"] == "note_midi":
#                     trk = []
#                     for note in d["data"]:
#                         trk.append({
#                                     "time": note["time"],
#                                     "duration": note["duration"],
#                                     "pitch": round(float(note["value"]))})
#                     if len(trk) > 0:
#                         midis.append({
#                             "source": d["annotation_metadata"]["data_source"],
#                             "instrument": "guitar",
#                             "notes": trk
#                         })
#             if len(midis) > 0:
#                 name = data["filename"].split("/")[-1]
#                 midi_path = os.path.join(output_folder, "midis")
#                 os.makedirs(midi_path, exist_ok=True)
#                 midi_path = os.path.join(midi_path, name + ".json")
#                 with open(midi_path, "w") as jsonfile:
#                     json.dump(midis, jsonfile, indent=2)
#                 data["midi"] = midi_path
#         data["instruments"] = "guitar"
#         data["monophonic ?"] = "no"
#         res.append(data)
#         print(data)


#     # pitch_contour
#     # note_midi
#     # beat_position
#     # tempo
#     # chord
#     # key_mode

#     with open(os.path.join(output_folder, "metadata.json"), 'w') as jsonfile:
#          json.dump(res, jsonfile, indent=2)

#     keys = {}
#     for data in res:
#         for key in data:
#             keys[key] = 0
#     with open(os.path.join(output_folder, "keys.lst"), "w") as f:
#         f.write("\n".join([k for k in keys]))

