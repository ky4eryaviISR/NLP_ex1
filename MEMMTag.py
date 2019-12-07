import copy
import operator
import pickle
import re
from datetime import datetime
from sys import argv
import numpy as np


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

words = {key for key in feature_file[0].split()}

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
    if word in words:
        feat["W_i"] = word
    else:
        feat["isDigit"] = any([character.isdigit() for character in word])
        feat["upperCase"] = any([character.isupper() for character in word])
        feat["hyphen"] = any([character == '-' for character in word])
        feat["suf_1"] = word[-1:]
        feat["suf_2"] = word[-2:]
        feat["suf_3"] = word[-3:]
        feat["suf_4"] = word[-4:]
        feat["pref_1"] = word[:1]
        feat["pref_2"] = word[:2]
        feat["pref_3"] = word[:3]
        feat["pref_4"] = word[:4]
    return feat



def get_argmax(stats):
    return max(stats.iteritems(), key=operator.itemgetter(1))[0]

def convert_2_vector(features):
    vector = np.zeros(len(label_2_index))
    for k, v in features.items():
        feature = k+"="+str(v)
        if feature in label_2_index.keys():
            vector[int(label_2_index[feature])] = 1
    return vector


labels = {k: int(v) for k, v in label_2_index.items() if '=' not in k}
labels_no_start = {k: int(v) for k, v in label_2_index.items() if '=' not in k and "START" not in k}
start = max(labels.values())+1
search_label = copy.deepcopy(labels)
labels["START"] = start
invert_labels = {v: k for k, v in label_2_index.items() if '=' not in k}
invert_labels[str(start)] = "START"
#cross_tags = np.full((len(labels), len(labels)), -np.inf)
cross_tags = {key: {value: -np.inf for value in labels.keys()}
               for key in labels.keys()}


def hmm_tag():
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
                           (re.compile("[A-Z]+$"), ['NNP', 'DT', 'PRP', 'JJ', 'NN',
                                                    'NNS', 'IN', 'NNP', 'TO', 'JJS',
                                                    'VBG', 'CC', 'VBD', 'PRP', 'RB',
                                                    'MD', 'VB', 'VBN', 'VBP', 'WDT',
                                                    'VBZ', 'WRB', '$', 'JJR', 'FW',
                                                    'RP', 'UH', 'CD', 'WP'])
                           ]
    for sentence in sentences[:10]:
        viterbi = [copy.deepcopy(cross_tags),copy.deepcopy(cross_tags)]
        viterbi[0]["START"]["START"] = 1
        tags = []
        w_p = w_pp = None
        w_n = sentence[1]
        w_nn = sentence[2]
        print(datetime.now())
        prev_tag_set = prev_prev_tag_set = ["START"]
        for j, word in enumerate(sentence):
            word_score = copy.deepcopy(cross_tags)
            best_tags = copy.deepcopy(cross_tags)
            if word in word_tag:
                possible_labels = word_tag[word]
            elif word.lower() in word_tag:
                possible_labels = word_tag[word.lower()]
            else:
                possible_labels = [labels for key, labels in regular_expressions if key.match(word)]
                possible_labels = possible_labels[0] if possible_labels else labels_no_start
            print(word)
            print(possible_labels)
            print(f"{datetime.now()}:RUN OVER TAGS")
            for t_second in prev_tag_set:
                best_score = {k: -np.inf for k in labels.keys()}
                score = {k: None for k in prev_tag_set}
                for t_first in prev_prev_tag_set:
                    features = get_features(word, t_p=t_second, t_pp=t_first, w_p=w_p, w_pp=w_pp, w_n=w_n, w_nn=w_nn)
                    new_score = model.predict_proba([convert_2_vector(features)])[0]
                    for label in possible_labels:
                        if new_score[labels[label]] > best_score[label]:
                            best_score[label] = new_score[labels[label]]
                            word_score[label][t_second] =  best_score[label]
                            best_tags[label][t_second] = t_first
            print(f"{datetime.now()}:FINISHED OVER TAGS")

            prev_prev_tag_set = prev_tag_set
            prev_tag_set = possible_labels
            w_pp, w_n = w_p, w_nn
            w_p = word
            w_nn = sentence[j+3] if len(sentence)-3 > j else None
            viterbi.append(word_score)
            tags.append(best_tags)

        cur_score = -np.inf
        last_tag = prev_tag = ""
        for prev, prev__prev_keys in viterbi[-1].items():
            for prev_prev, score in prev__prev_keys.items():
                if cur_score < score:
                    cur_score = score
                    prev_tag = prev_prev
                    last_tag = prev
        sen_len=len(sentence)
        tag_s = [None]*sen_len
        tag_s[-1] = last_tag
        tag_s[-2] = prev_tag
        for index in range(len(tags)-1, 1, -1):
            t_tag = tags[index][last_tag][prev_tag]
            tag_s[index-2] = t_tag
            last_tag = prev_tag
            prev_tag = t_tag
        #print(' '.join([s+"/"+t for s, t in zip(sentence,tag_s)]))
        #print(i)
    #return sentences[start: stop]

hmm_tag()