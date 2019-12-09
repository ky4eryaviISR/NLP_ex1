import copy
import pickle
import re
from functools import reduce
from sys import argv

import math
import numpy as np
from datetime import datetime
import multiprocessing as mp

from calculate_accuracy import calculate_accuracy


def print_to_file(result, output_file):
    with open(output_file, "w") as f:
        for sentence in result:
            output = ""
            for tag, word in sentence:
                output += word + '/' + tag + ' '
            f.write(output + '\n')

def load_input_file():
    data = []
    with open(input_file) as f:
        for line in f:
            data.append([w for w in line.split()])
    return data


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

def create_word_tag(feature):
    word_tag = {}
    for i in feature.split(' ')[:-1]:
        word = i.split('~')[0]
        word_tag[word] = set()
        for tag in i.split('~', )[1].split('/'):
            word_tag[word].add(tag)
    return word_tag

def convert_2_vector(features):
    vector = np.zeros(len(label_2_index))
    for k, v in features.items():
        feature = k+"="+str(v)
        if feature in label_2_index.keys():
            vector[int(label_2_index[feature])] = 1
    return vector


def memm_tag(sentences, model, label_2_index, word_tag,words):
    results = []
    labels = {k: int(v) for k, v in label_2_index.items() if '=' not in k}
    labels_no_start = copy.copy(labels)
    invert_labels = {v: k for k, v in label_2_index.items() if '=' not in k}
    labels["START"] = max(labels.values()) + 1
    for sentence in sentences:
        # initialize start of the sentence
        viterbi = np.full((len(labels), len(labels)), -np.inf)
        tags = []
        viterbi[labels["START"], labels["START"]] = 1

        w_p = w_pp = None
        w_n = sentence[1] if len(sentence) > 1 else None
        w_nn = sentence[2] if len(sentence) > 2 else None
        p_set_tag = pp_set_tag = ["START"]
        pp_set_tag = {i: labels[i] for i in pp_set_tag}
        # run over each word
        for i, word in enumerate(sentence):

            vit = np.full((len(labels), len(labels)), -np.inf)
            best_tags = np.full((len(labels), len(labels)), -np.inf)
            t_word = word.lower()
            if word in words:
                possible_labels = word_tag[word]
            elif t_word in words:
               possible_labels = word_tag[t_word]
            else:
                possible_labels = [labels for key, labels in regular_expressions if key.match(word)]
                possible_labels = possible_labels[0] if possible_labels else labels_no_start
            p_set_tag = {i: labels[i] for i in p_set_tag}
            for t_p, t_p_code in p_set_tag.items():
                features = np.zeros((len(pp_set_tag),len(label_2_index)))
                keys = np.zeros(len(pp_set_tag))
                v_temp = np.zeros((len(pp_set_tag), len(labels_no_start)))
                v_cnt = 0
                for t_pp, t_pp_code in pp_set_tag.items():
                    vec = convert_2_vector(get_features(word, t_p=t_p, t_pp=t_pp, w_p=w_p,
                                                        w_pp=w_pp, w_n=w_n, w_nn=w_nn))
                    features[v_cnt] = vec
                    keys[v_cnt] = t_pp_code
                    v_temp[v_cnt, :] = viterbi[t_p_code][t_pp_code]
                    v_cnt += 1

                predicted = v_temp + model.predict_proba(features)
                for lbl in possible_labels:
                    lbl_code = labels[lbl]
                    tag_value = predicted[:, lbl_code]
                    vit[lbl_code, t_p_code] = max(tag_value)
                    best_tags[lbl_code][t_p_code] = keys[int(np.argmax(tag_value))]
            pp_set_tag = p_set_tag
            p_set_tag = possible_labels
            w_pp, w_n = w_p, w_nn
            w_p = word
            w_nn = sentence[i + 3] if len(sentence) - 3 > i else None
            viterbi = vit
            tags.append(best_tags)

        cur_score = -np.inf
        last_tag = prev_tag = ""
        for i in range(viterbi.shape[0]):
            for j in range(viterbi.shape[0]):
                if cur_score < viterbi[i][j]:
                    cur_score = viterbi[i][j]
                    prev_tag = j
                    last_tag = i

        sen_len = len(sentence)
        tag_s = [None] * sen_len
        tag_s[-1] = invert_labels[last_tag]
        sen = [[tag_s[-1], sentence[-1]]]
        if sen_len != 1:
            tag_s[-2] = invert_labels[prev_tag]
            sen.append([tag_s[-2], sentence[-2]])
        i = 3
        for index in range(len(tags) - 1, 1, -1):
            new_tag = int(tags[index][last_tag][prev_tag])
            last_tag = prev_tag
            prev_tag = new_tag
            sen.append([invert_labels[new_tag], sentence[-i]])
            i += 1
        results.append(reversed(sen))
    return results


input_file = argv[1]
model = argv[2]
feature_file = argv[3]
out_file = argv[4]
feature_file = open(feature_file).read().split('\n')
label_2_index = {line.split()[0]: int(line.split()[1]) for line in feature_file[3:]}
index_2_label = {line.split()[1]: line.split()[0] for line in feature_file[3:]}
regular_expressions = [[re.compile(line.split('~',1)[0]),[item for item in line.split('~',1)[1].split('/')]] for line in feature_file[2].split()]
words = [word for word in feature_file[1].split()]
sentences = load_input_file()
word_tag = create_word_tag(feature_file[0])

model = pickle.load(open(model, 'rb'))


if __name__ == "__main__":
    start = datetime.now()
    print(f"{datetime.now()}: START")

    sentences = sentences
    num_sentences = len(sentences)
    steps = math.ceil((num_sentences / mp.cpu_count()))
    args = []
    for i in range(0, num_sentences, steps):
        args.append([sentences[i:i+steps], model, label_2_index, word_tag, words])

    pool = mp.Pool(mp.cpu_count())
    results = pool.starmap(memm_tag, args)
    results = reduce(lambda x, y: x + y, results)
    print_to_file(results, out_file)
    print(f"{datetime.now()}: START")




