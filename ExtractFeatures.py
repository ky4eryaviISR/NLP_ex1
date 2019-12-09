from sys import argv
from datetime import datetime



WORD = 1
TAG = 0


def load_input_file(input_file):
    data = []
    word_dic = {}
    with open(input_file) as f:
        for line in f:
            sentence_formatted = [["START", None], ["START", None]]
            for part in line.split():
                word, tag = part.rsplit('/', 1)
                sentence_formatted.append([tag, word])
                word_dic[word] = 1 if word not in word_dic else word_dic[word] + 1
                # store all tags for word
            data.append(sentence_formatted)
    return data, word_dic


def create_features():
    features = []
    for sentence in sentences:
        for i in range(2, len(sentence)):
            feat = {}
            word = sentence[i][WORD]
            if word_count[word] < 5:
                feat["isDigit"] = any([character.isdigit() for character in word])
                feat["upperCase"] = any([character.isupper() for character in word])
                feat["hyphen"] = any([character == '-' for character in word])
                for j in range(1, 5, 1):
                    if len(word) >= j:
                        feat["suf_"+str(j)] = word[-j:]
                        feat["pref_"+str(j)] = word[:j]
            else:
                feat["W_i"] = word
            feat["t_i_prev"] = sentence[i - 1][TAG]
            feat["t_i_prev_prev"] = sentence[i - 2][TAG] + "_" + sentence[i - 1][TAG]
            feat["w_i_prev"] = sentence[i - 1][WORD] if i > 2 else None
            feat["w_i_prev_prev"] = sentence[i - 2][WORD] if i > 3 else None
            feat["w_i_next"] = sentence[i + 1][WORD] if i < len(sentence) - 1 else None
            feat["w_i_next_next"] = sentence[i + 2][WORD] if i < len(sentence) - 2 else None

            features.append(sentence[i][TAG] + ' ' + " ".join([f"{k}={v}" for k, v in feat.items()]))
    return features


def print_to_file():
    with open(out_file, 'w') as f:
            f.write("".join([s+"\n" for s in feat_sentences]))


start = datetime.now()
input_file = argv[1]
out_file = argv[2]

sentences, word_count = load_input_file(input_file)
feat_sentences = create_features()
print_to_file()
