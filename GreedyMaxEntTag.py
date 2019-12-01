from datetime import datetime
import pickle
from sys import argv

from calculate_accuracy import calculate_accuracy

TAG = 0
WORD = 1
start = datetime.now()
print(start)


def print_to_file(result, output_file):
    with open(output_file, "w") as f:
        for sentence in result:
            output = ""
            for tag, word in sentence:
                output += word + '/' + tag + ' '
            f.write(output + '\n')


def convert_2_vector():
    vector = [0]*(len(label_2_index.keys()))
    for k,v in features.items():
        feature = k+"="+str(v)
        if feature in label_2_index.keys():
            vector[int(label_2_index[feature])] = 1

    return vector


def load_input_file():
    data = []
    with open(input_file) as f:
        for line in f:
            data.append([["START", "*"], ["START", "*"]] + [['*', w] for w in line.split()])
    return data


def get_features(i):
    word = sentence[i][WORD]
    feat = {"isDigit": any([character.isdigit() for character in word]),
            "upperCase": any([character.isupper() for character in word]),
            "hyphen": any([character == '-' for character in word]),
            "W_i": word,
            "t_i_prev": sentence[i - 1][TAG],
            "t_i_prev_prev": sentence[i - 2][TAG] + "_" + sentence[i - 1][TAG]
            }
    if i < len(sentence) - 2:
        feat["w_i_next"] = sentence[i + 1][WORD] if i < len(sentence) - 1 else "*"
    if i < len(sentence) - 3:
        feat["w_i_next_next"] = sentence[i + 2][WORD] if i < len(sentence) - 2 else "*"
    if i > 2:
        feat["w_i_prev"] = sentence[i - 1][WORD]
    if i > 3:
        feat["w_i_prev_prev"] = sentence[i - 2][WORD]
    for j in range(1, 5, 1):
        feat["suf_" + str(j)] = word[-j:]
        feat["pref_" + str(j)] = word[:j]
    return feat


if __name__ == '__main__':
    input_file = argv[1]
    model = argv[2]
    feature_file = argv[3]
    out_file = argv[4]

    sentences = load_input_file()
    feature_file = open(feature_file).read().split('\n')
    label_2_index = {line.split()[0]:line.split()[1] for line in feature_file}
    index_2_label = {line.split()[1]: line.split()[0] for line in feature_file}
    model = pickle.load(open(model, 'rb'))

    for sentence in sentences:
        for i in range(2, len(sentence)):
            features = get_features(i)
            vector = convert_2_vector()
            tag_predicted = model.predict([vector])
            sentence[i][TAG] = index_2_label[str(int(tag_predicted[0]))]

    print_to_file(sentences, out_file)
    calculate_accuracy(out_file, 'data/ass1-tagger-dev')
    print(datetime.now() - start)