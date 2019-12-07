import copy
import operator
import pickle
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

words = {key for key in feature_file[1].split()}

word_tag = {}
for i in feature_file[2].split(' ')[:-1]:
    word = i.split(':')[0]
    word_tag[word] = {}
    for tag in i.split(':', 1)[1].split('/'):
        t_pp, t_p = tag.split('_')
        if t_p not in word_tag[word]:
            word_tag[word][t_p] = []
        word_tag[word][t_p].append(t_pp)

word_tag['None'] = ["START"]
label_2_index = {line.split()[0]: line.split()[1] for line in feature_file[3:]}
index_2_label = {line.split()[1]: line.split()[0] for line in feature_file[3:]}
model = pickle.load(open(model, 'rb'))

labels = {k: int(v) for k, v in label_2_index.items() if '=' not in k}
start = max(labels.values())+1
search_label = copy.deepcopy(labels)
labels["START"] = start
invert_labels = {v: k for k, v in label_2_index.items() if '=' not in k}
invert_labels[str(start)] = "START"



cross_tags = np.full((len(labels), len(labels)), -np.inf)
#cross_tags = {key: {value: -np.inf for value in labels.keys()}
              # for key in labels.keys()}

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


bi_tag = {}
for part in feature_file[0].split():
    t, t_n = part.split('_')
    if t not in bi_tag:
        bi_tag[t] = []
    bi_tag[t].append(t_n)




def get_argmax(stats):
    return max(stats.iteritems(), key=operator.itemgetter(1))[0]

def convert_2_vector(features):
    vector = np.zeros(len(label_2_index))
    for k, v in features.items():
        #print(f"{datetime.now() - start}: LOOP")
        feature = k+"="+str(v)
        if feature in label_2_index.keys():
            vector[int(label_2_index[feature])] = 1
    return vector




def hmm_tag():
    for sentence in sentences:
        #viterbi = [copy.deepcopy(cross_tags)]
       # viterbi[0][labels["START"]][labels["START"]] = 1
        tags = []
        w_p = w_pp = None
        w_n = sentence[1]
        w_nn = sentence[2]
        print(datetime.now())

        vec_i = 0
        for j, word in enumerate(sentence):
            word_score = copy.deepcopy(cross_tags)
            best_tags = copy.deepcopy(cross_tags)
            if word in word_tag and w_p in word_tag:
                possible_tags = set(word_tag[word].keys).intersection(word_tag[w_p])
            if word in word_tag:
                possible_tags = word_tag[word]
            else:
                possible_tags = labels

            vectors_input = []
            for t_second in possible_tags.keys():
                t_sec_code = labels[t_second]
                #print(f"{datetime.now()}:if word in word_tag choose tag")
                if word in word_tag:
                    possible_first_tag = list(word_tag[word][t_second])
                else:
                    possible_first_tag = bi_tag[t_second]
                for t_first in possible_first_tag:
                    t_first_code = labels[t_first]
                    #print(f"{datetime.now()}:get features")
                    features = get_features(word, t_p=t_second, t_pp=t_first, w_p=w_p, w_pp=w_pp, w_n=w_n, w_nn=w_nn)
                    #print(f"{datetime.now()}:convert_vector")
                    vectors_input.append(convert_2_vector(features))
                    #print(f"{datetime.now()}:predict")

            new_score = model.predict_proba(vectors_input)


                    # new_score = model.predict_proba([vector])
                    #print(f"{datetime.now()}:start run over results")




#                     for tag in search_label.values():
#                         score = new_score[0][tag] + viterbi[-1][t_sec_code][t_first_code]
#                         if score > word_score[tag][t_sec_code]:
# #                            print(f"TAG:{tag} PREV:{t_first}_{t_second}")
# #                            print(f"PREV_TAG:{best_tags[tag][t_second]} SCORE:{word_score[tag][t_second]}")
#                             word_score[tag][t_sec_code] = score
#                             best_tags[tag][t_sec_code] = t_first_code
#                            print(f"NEW_TAG:{best_tags[tag][t_second]} NEW SCORE:{word_score[tag][t_second]}\n\n")
                    #print(f"{datetime.now()}:stop run over results")
            #viterbi.pop(0)
            # viterbi.append(word_score)
            # tags.append(best_tags)
            w_pp, w_n = w_p, w_nn
            w_p = word
            w_nn = sentence[j+3] if len(sentence)-3 > j else None

        print(datetime.now())
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
        tag_s[-1]=last_tag
        tag_s[-2]=prev_tag
        for index in range(len(tags)-1, 1, -1):
            t_tag = tags[index][last_tag][prev_tag]
            tag_s[index-2] = t_tag
            last_tag = prev_tag
            prev_tag = t_tag
        print(' '.join([s+"/"+t for s, t in zip(sentence,tag_s)]))
        #print(i)
    #return sentences[start: stop]

hmm_tag()