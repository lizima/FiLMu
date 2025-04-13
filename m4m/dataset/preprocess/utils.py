from music21 import converter


def parse_xml(path):
    score = converter.parse(path)
    return [[part.id, part] for part in score.parts]


def is_swing_tempo(path):
    swing_keywords = ["swing", "shuffle", "Swing", "Shuffle"]  # Common keywords indicating swing tempo
    with open(path, "r") as f:
        line = f.read()
    for w in swing_keywords:
        if len(line.split(w)) == 2:
            return "yes"
    return "no"


def get_beats_by_measure(score):

    time_signatures = score.flatten().getElementsByClass('TimeSignature')

    beats_by_measure = []
    current_measure = 0

    # Iterate through time signatures
    for ts in time_signatures:
        measure_number = ts.measureNumber

        # Fill in any missing measures with the previous time signature's beats
        for i in range(current_measure + 1, measure_number):
            beats_by_measure.append(ts.beatCount)

        # Add the current time signature's beats
        beats_by_measure.append(ts.beatCount)

        current_measure = measure_number

    # If the piece ends before the last time signature, fill in remaining measures
    num_measures = 0
    for element in score.recurse():
        if 'Measure' in element.classes:
            num_measures += 1

    if current_measure < num_measures:
        last_time_signature = time_signatures[-1]
        for i in range(current_measure + 1, num_measures + 1):
            beats_by_measure.append(last_time_signature.beatCount)

    return beats_by_measure


def parse_score(score):

    # Extract time signature
    time_signature = score.flatten().getElementsByClass('TimeSignature')
    beats = []


    for ts in time_signature:
        b = ts.numerator
        beat_type = ts.denominator

        beats.append(str(b) + "/" + str(beat_type))


    if len(beats) == 1:
        beats = beats[0]
    else:
        beats = " - ".join(beats)

    # Extract key signature
    key_modes = []
    key_signature = score.flatten().getElementsByClass('KeySignature')
    for ks in key_signature:
        ks = ks.getScale()
        key_mode = ks.name
        key_mode = str.replace(key_mode, "+", "#")
        key_mode = str.replace(key_mode, "-", "b")
        key_mode = str.replace(key_mode, " ", ":")
        key_modes.append(key_mode)
    if len(key_modes) == 1:
        key_modes = key_modes[0]
    else:
        key_modes = " - ".join(key_modes)

    tempos = score.flatten().getElementsByClass('MetronomeMark')

    tempos = " - ".join([str(t.number) for t in tempos])


    return beats, key_modes, tempos
