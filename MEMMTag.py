import copy
import operator
import pickle
import re
from datetime import datetime, time
from sys import argv
import numpy as np

from calculate_accuracy import calculate_accuracy


regular_expressions = [(re.compile('\w+ness$'),['NN','NNP','VB']),
                           (re.compile("^\d+\.?\d+$"),['CD','NNP','VBN']),
                           (re.compile("\w+ing$"),['NN','VBG','JJ','IN','NNP','VB','VBP','RB']),
                           (re.compile("\w+s$"),['POS','VBZ','PRP','NNP','NNS']),
                           (re.compile("\w+ion$"),['NNP','NN','CD','VB','VBP','JJ','FW',',']),
                           (re.compile("\w+al$"),['NN','JJ','NNP','RB','VB','VBP','IN']),
                           (re.compile("\w+-\w+$"), ['NNP', 'JJ', 'NN', 'VBG', 'VBN', 'NNS', 'RB', 'JJR', 'VBD',
                                                     'CD', 'JJS', 'VB', 'VBP', 'NNP', 'UH', 'RBR', 'VBZ', 'FW']),
                           (re.compile("'\w+$"), ['POS', 'VBZ', 'VBP', 'MD', 'VBD', 'PRP',
                                                  'VB', 'NNP', 'CD', 'IN', 'NNS', 'CC']),
                           (re.compile("\w+ed$"), ['VBN', 'VBD', 'JJ', 'NNP', 'RB', 'NN', 'CD',
                                                   'VBP', 'VB', 'UH', 'VBG', 'MD', 'NNS', 'RBR']),
                            (re.compile("\w+er$"),['NN', 'NNP', 'JJ', 'RBR', 'RB', 'IN',
                                              'JJR', 'VB', 'RP', 'PRP$', 'DT', 'VBP',
                                              'CC', 'PRP', 'WDT']),
                           (re.compile("[A-Z]+$"), ['NNP', 'DT', 'PRP', 'JJ', 'NN',
                                                    'NNS', 'IN', 'NNP', 'TO', 'JJS',
                                                    'VBG', 'CC', 'VBD', 'PRP', 'RB',
                                                    'MD', 'VB', 'VBN', 'VBP', 'WDT',
                                                    'VBZ', 'WRB', '$', 'JJR', 'FW',
                                                    'RP', 'UH', 'CD', 'WP'])
                           ]

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


input_file = argv[1]
model = argv[2]
feature_file = argv[3]
out_file = argv[4]

sentences = load_input_file()
feature_file = open(feature_file).read().split('\n')

word_tag = {}
for i in feature_file[0].split(' ')[:-1]:
    word = i.split(':')[0]
    word_tag[word] = set()
    for tag in i.split('::', )[1].split('/'):
        word_tag[word].add(tag)

word_tag['None'] = ["START"]
label_2_index = {line.split()[0]: int(line.split()[1]) for line in feature_file[1:]}
index_2_label = {line.split()[1]: line.split()[0] for line in feature_file[1:]}
model = pickle.load(open(model, 'rb'))






def get_features(word, t_p, t_pp, w_p, w_pp, w_n, w_nn):
    feat = {"t_i_prev": t_p,
            "t_i_prev_prev": t_p + "_" + t_pp,
            "w_i_next": w_n,
            "w_i_next_next": w_nn,
            "w_i_prev": w_p,
            "w_i_prev_prev": w_pp
            }
    if word in word_tag:
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

def convert_2_vector(features):
    vector = np.zeros(len(label_2_index))
    for k, v in features.items():
        feature = k+"="+str(v)
        if feature in label_2_index.keys():
            vector[int(label_2_index[feature])] = 1
    return vector




def hmm_tag():
    labels = {k: int(v) for k, v in label_2_index.items() if '=' not in k}
    labels_no_start = {k: int(v) for k, v in label_2_index.items() if '=' not in k and "START" not in k}
    start = max(labels.values()) + 1
    labels["START"] = start
    invert_labels = {v: k for k, v in label_2_index.items() if '=' not in k}
    invert_labels[str(start)] = "START"
    results = []
    ss=datetime.now()
    i_num = 0
    for sentence in sentences[:2]:
        print(f"******************************{i_num} {datetime.now() - ss}******************************")
        i_num += 1
        ss = datetime.now()
        viterbi = np.full((len(labels),len(labels)),-np.inf)
        viterbi[labels["START"]][labels["START"]] = 1
        tags = []
        w_p = w_pp = None
        w_n = sentence[1]
        w_nn = sentence[2]
        prev_tag_set = prev_prev_tag_set = ["START"]
        for j, word in enumerate(sentence):
            vit = np.full((len(labels),len(labels)),-np.inf)
            #print(f"{datetime.now() - ss}:START NEW WORD")
            word_score = np.full((len(labels), len(labels)), -np.inf)
            best_tags = np.full((len(labels), len(labels)), -np.inf)
            if word in word_tag.keys():
                possible_labels = word_tag[word]
            elif word.lower() in word_tag.keys():
                possible_labels = word_tag[word.lower()]
            else:
                possible_labels = [labels for key, labels in regular_expressions if key.match(word)]
                possible_labels = possible_labels[0] if possible_labels else labels_no_start
            #print(word)

            # print(f"{datetime.now()-ss}:RUN OVER TAGS")
            # print(possible_labels)
            # print(prev_tag_set)
            # print(prev_prev_tag_set)
            for t_second in prev_tag_set:
                t_second_code = labels[t_second]
                best_score = np.full((len(invert_labels)),-np.inf)
                temp = {}
                for lbl in possible_labels:
                    lbl_code = labels[lbl]
                    temp[lbl_code] = {}

                for t_first in prev_prev_tag_set:
                    t_first_code = labels[t_first]
                    features = get_features(word, t_p=t_second, t_pp=t_first, w_p=w_p, w_pp=w_pp, w_n=w_n, w_nn=w_nn)
                    # print(f"{datetime.now() - ss}:CALCULATE PREDICT")
                    new_score = viterbi[t_second_code][t_first_code] + model.predict_proba([convert_2_vector(features)])[0]
                    # print(f"{datetime.now() - ss}:FINISHED CALCULATE PREDICT")
                    # print(f"{datetime.now() - ss}:CALCULATE VECTORS")
                    for lbl in possible_labels:
                        lbl_code = labels[lbl]
                        temp[lbl_code][t_first_code] = new_score[lbl_code]
                for lbl in possible_labels:
                    lbl_code = labels[lbl]
                    vit[t_second_code][lbl_code] = max(list(temp[lbl_code].values()))
                    best_tags[t_second_code][lbl_code] = max(temp[lbl_code], key=temp[lbl_code].get)
            # print(f"{datetime.now()-ss}:FINISHED OVER TAGS")

            prev_prev_tag_set = prev_tag_set
            prev_tag_set = possible_labels
            w_pp, w_n = w_p, w_nn
            w_p = word
            w_nn = sentence[j+3] if len(sentence)-3 > j else None
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
        sen_len=len(sentence)
        tag_s = [None]*sen_len
        tag_s[-1] = invert_labels[last_tag]
        tag_s[-2] = invert_labels[prev_tag]
        sen = [[tag_s[-1], sentence[-1]], [tag_s[-2], sentence[-2]]]
        i = 3
        for index in range(len(tags) - 1, 1, -1):
            new_tag = int(tags[index][last_tag][prev_tag])
            last_tag = prev_tag
            prev_tag = new_tag
            sen.append([invert_labels[new_tag], sentence[-i]])
            i += 1
        results.append(reversed(sen))
    return results

results = hmm_tag()
print_to_file(results, out_file)
calculate_accuracy('out_file', 'data/ass1-tagger-dev')