from datetime import datetime
import pickle
from functools import reduce
from sys import argv
import multiprocessing as mp
import math
import numpy as np

from calculate_accuracy import calculate_accuracy

def load_input_file():
    data = []
    with open(input_file) as f:
        for line in f:
            data.append([w for w in line.split()])
    return data



input_file = argv[1]
model = argv[2]
feature_file = argv[3]
out_file = argv[4]

sentences = load_input_file()
feature_file = open(feature_file).read().split('\n')

words = [word for word in feature_file[1].split()]

label_2_index = {line.split()[0]: line.split()[1] for line in feature_file[3:]}
index_2_label = {line.split()[1]: line.split()[0] for line in feature_file[3:]}
model = pickle.load(open(model, 'rb'))

def print_to_file(result, output_file):
    with open(output_file, "w") as f:
        for sentence in result:
            output = ""
            for tag, word in sentence:
                output += word + '/' + tag + ' '
            f.write(output + '\n')


def convert_2_vector(features):
    vector = np.zeros(len(label_2_index))
    for k, v in features.items():
        feature = k+"="+str(v)
        if feature in label_2_index.keys():
            vector[int(label_2_index[feature])] = 1
    return vector


def get_features(word, t_p, t_pp, w_p, w_pp, w_n, w_nn):
    feat = {"t_i_prev": t_p,
            "t_i_prev_prev": t_p + "_" + t_pp,
            "w_i_next": w_n,
            "w_i_next_next": w_nn,
            "w_i_prev": w_p,
            "w_i_prev_prev": w_pp
            }
    if word in words:
        feat["W_i"] = word
    else:
        feat["isDigit"] = any([character.isdigit() for character in word])
        feat["upperCase"] = any([character.isupper() for character in word])
        feat["hyphen"] = any([character == '-' for character in word])
        for j in range(1, 5, 1):
            if len(word) >= j:
                feat["suf_" + str(j)] = word[-j:]
                feat["pref_" + str(j)] = word[:j]
    return feat


def memm_greedy(sentences):
    res = [[]]*len(sentences)
    for i, sentence in enumerate(sentences):
        t_p = t_pp = "START"
        w_p = w_pp = None
        w_n = sentence[1] if len(sentence) > 1 else None
        w_nn = sentence[2] if len(sentence) > 2 else None
        sent_list = list()
        for j in range(len(sentence)):
            features = get_features(sentence[j], t_p, t_pp, w_p, w_pp, w_n, w_nn)
            vector = convert_2_vector(features)
            tag_predicted = model.predict([vector])
            t_pp, w_pp, w_n = t_p, w_p, w_nn
            w_p = sentence[j]
            w_nn = sentence[j+3] if len(sentence)-3 > j else None
            t_p = index_2_label[str(int(tag_predicted[0]))]
            sent_list.append((w_p+"/"+t_p))
        res[i] = sent_list
    return res


if __name__ == '__main__':
    start = datetime.now()
    print(start)
    results = memm_greedy(sentences)
    with open(out_file, 'w') as f:
        f.write('\n'.join(' '.join(map(str, row)) for row in results))
    print(datetime.now() - start)
