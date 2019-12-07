from datetime import datetime
import pickle
from multiprocessing import cpu_count
from sys import argv

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression

start = datetime.now()
print(start)


def replace_labels():
    word_prev_tags = {}
    vectors = []
    for sentence in feat_sentences:
        label = [str(feature_labels[sentence.split()[0]])]
        sorted_features = sorted([feature_labels[word] for word in sentence.split()[1:]])
        vector = label + [str(feat) + ":1" for feat in sorted_features]
        vectors.append(vector)
        find = sentence.find('W_i')
        if find != -1:
            word = sentence[find:].split(' ', 1)[0].split('=')[1]
            find = sentence.find('t_i_prev_prev=')
            t_prev_prev = sentence[find:].split(' ', 1)[0].split('=')[1]
            if word not in word_prev_tags:
                word_prev_tags[word] = set()
            word_prev_tags[word].add(t_prev_prev)

    with open("features_vec_file", "w") as f_result:
        f_result.write('\n'.join(' '.join(map(str, sl)) for sl in vectors))
    with open('features_map_file', "w") as f:
        f.write(" ".join([key[14:] for key in feature_labels.keys() if key.startswith('t_i_prev_prev')]))
        f.write('\n')
        f.write(" ".join([key[4:] for key in feature_labels.keys() if key.startswith('W_i')]))
        f.write('\n')
        f.write(' '.join([key+":"+"/".join([tag for tag in value]) for key,value in word_prev_tags.items()]))
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

x_train, y_train = load_svmlight_file("features_vec_file", zero_based=True)
for sol in ['saga']:
    print(sol)
    print(datetime.now())
    model = LogisticRegression(multi_class='auto', solver='saga',#n_jobs=cpu_count(),# solver=sol, multi_class='multinomial',
                               tol=0.001)#, penalty='l2', max_iter=150)
    model.fit(x_train, y_train)
    print(model.score(x_train, y_train))
    print(datetime.now())


pickle.dump(model, open(out_file, 'wb'))
