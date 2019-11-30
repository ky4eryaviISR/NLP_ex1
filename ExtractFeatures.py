from sys import argv

WORD = 1
TAG = 0

def load_input_file(input_file):
    data = []
    word_dic = {}
    with open(input_file) as f:
        for line in f:
            sentence_formatted = [["START", "*"], ["START", "*"]]
            for part in line.split():
                word, tag = part.rsplit('/',1)
                sentence_formatted.append([tag, word])
                word_dic[word] = 1 if word not in word_dic else word_dic[word] + 1
            data.append(sentence_formatted)
    return data, word_dic

def create_features():
    features = []
    for sentence in sentences:
        for i in range(2,len(sentence)):
            feat = {}
            word = sentence[i][WORD]
            if words[word] < 5:
                feat["isDigit"] = any([character.isdigit() for character in word])
                feat["upperCase"] = any([character.isupper() for character in word])
                feat["upperCase"] = any([character == '-' for character in word])
                feat["suf_1"] = word[-1:]
                feat["suf_2"] = word[-2:]
                feat["suf_3"] = word[-3:]
                feat["suf_4"] = word[-4:]
                feat["pref_1"] = word[:1]
                feat["pref_2"] = word[:2]
                feat["pref_3"] = word[:3]
                feat["pref_4"] = word[:4]
            else:
                feat["W_i"] = word
            feat["t_i_prev"] = sentence[i - 1][TAG]
            feat["t_i_prev_prev"] = sentence[i - 2][TAG] + " " + sentence[i - 1][TAG]
            feat["w_i_prev"] = sentence[i - 1][WORD]
            feat["w_i_prev_prev"] = sentence[i - 2][WORD]
            feat["w_i_next"] = sentence[i + 1][WORD] if i < len(sentence)-1 else "*"
            feat["w_i_next_next"] = sentence[i + 2][WORD] if i < len(sentence)-2 else "*"
            features.append([sentence[i][TAG], feat])
    return features

def print_to_file(features):
    with open(out_file, 'w') as f:
        for feature in features:
            str_out = feature[0]+" "+" ".join([f"{k}={str(v)}" for k, v in feature[1].items()])
            f.write(str_out+"\n")

input_file = argv[1]
out_file = argv[2]
sentences, words = load_input_file(input_file)
create_features()
print_to_file(create_features())

print("")

