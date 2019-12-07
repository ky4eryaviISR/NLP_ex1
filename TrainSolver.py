from datetime import datetime
import pickle
from multiprocessing import cpu_count
from sys import argv

from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import LogisticRegression

start = datetime.now()
print(start)

def replace_labels():
    with open("features_vec_file", "w") as f_result:
        for sentence in feat_sentences:
            label = [str(feature_labels[sentence.split()[0]])]
            sorted_features = sorted([feature_labels[word] for word in sentence.split()[1:]])
            vector = label + [str(feat) + ":1" for feat in sorted_features]
            f_result.write(" ".join(vector) + "\n")

    with open('features_map_file', "w") as f:
        f.write("\n".join([f"{key} {value}" for key, value in feature_labels.items()]))

#
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
                               tol=0.01)#, penalty='l2', max_iter=150)
    model.fit(x_train, y_train)
    print(model.score(x_train, y_train))
    print(datetime.now()-start)


pickle.dump(model, open(out_file, 'wb'))