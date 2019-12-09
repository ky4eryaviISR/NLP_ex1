import re
from datetime import datetime
import pickle
from sys import argv

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression
regex = [re.compile('\w+ness$'),
         re.compile("^\d+\.?\,?\:?\-?\/?\d+$"),
         re.compile("\w+ing$"),
         re.compile("\w+s$"),
         re.compile("\w+ec$"),
         re.compile("\w+ion$"),
         re.compile("\w+al$"),
         re.compile('[A-Z]+[a-z]+$'),
         re.compile("\w+-\w+$"),
         re.compile("'\w+$"),
         re.compile("\w+ed$"),
         re.compile("\w+er$"),
         re.compile("[A-Z]+$"),
         re.compile("\w+able$"),
         re.compile("\w+ve$"),
         re.compile("\w+tic$"),
         re.compile("\w+-\w+-\w+$"),
         re.compile("\w+ose$")]

def replace_labels():
    word_prev_tags = {}
    vectors = []
    word_count = {}
    for sentence in feat_sentences:
        label = [str(feature_labels[sentence.split()[0]])]
        sorted_features = sorted([feature_labels[word] for word in sentence.split()[1:]])
        vector = label + [str(feat) + ":1" for feat in sorted_features]
        vectors.append(vector)
        find = sentence.find('w_i_prev=')
        word = sentence[find:].split(' ', 1)[0].split('=')[1]
        find = sentence.find('t_i_prev=')
        t_prev = sentence[find:].split(' ', 1)[0].split('=')[1]
        if word == 'None' and t_prev=='START':
            continue
        if word not in word_prev_tags:
            word_prev_tags[word] = set()
        word_prev_tags[word].add(t_prev)
        word_count[word] = 1 if word not in word_count else word_count[word] + 1

    reg_dict = {}
    for reg in regex:
        for word in word_prev_tags.keys():
            if word !='None' and reg.match(word):
                if reg.pattern not in reg_dict:
                    reg_dict[reg.pattern] = set()
                for item in word_prev_tags[word]:
                    reg_dict[reg.pattern].add(item)


    with open("feature_vec_file", "w") as f_result:
        f_result.write('\n'.join(' '.join(map(str, sl)) for sl in vectors))
    with open('feature_map_file', "w") as f:
        f.write(' '.join([key+"~"+"/".join([tag for tag in value]) for key, value in word_prev_tags.items()]))
        f.write('\n')
        f.write(' '.join([word for word, cnt in word_count.items() if cnt >= 5]))
        f.write('\n')
        f.write(' '.join([pat+'~'+'/'.join([item for item in reg_dict[pat]]) for pat in reg_dict]))
        f.write('\n')
        f.write("\n".join([f"{key} {value}" for key, value in feature_labels.items()]))


def get_features():
    features_labels = {}
    i = 0

    for label in tags:
        features_labels[label] = i
        i += 1

    for feature_sentence in feat_sentences:
        for feature in feature_sentence.split():
            if feature not in features_labels:
                features_labels[feature] = i
                i += 1

    return features_labels


input_file = argv[1]
out_file = argv[2]
feat_sentences = [line[:-1] for line in open(input_file)]
tags = set([sentence.split()[0] for sentence in feat_sentences])
feature_labels = get_features()
replace_labels()

x_train, y_train = load_svmlight_file("feature_vec_file", zero_based=True)
print(datetime.now())
model = LogisticRegression(multi_class='auto', solver='sag', tol=0.001)
model.fit(x_train, y_train)
print(model.score(x_train, y_train))
print(datetime.now())


pickle.dump(model, open(out_file, 'wb'))
