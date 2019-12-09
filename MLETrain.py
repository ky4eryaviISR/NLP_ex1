from sys import argv
import Utils


def parse_file(input_file, e_output, q_output):
    lines = [line.split() for line in open(input_file)]
    word_dict = {"*UNK*": {}
                 }
    tag_dict = {"START": 0,
                "START START": 0}
    for line in lines:
        first = second = "START"
        tag_dict["START"] += 2
        tag_dict["START START"] += 1
        for third in line:
            word, tag = third.rsplit('/', 1)
            tag_store = tag
            # storing uni, bi and tri grams by tag
            tag_dict[tag] = 1 if tag not in tag_dict.keys() else tag_dict[tag] + 1
            tag = second + " " + tag
            tag_dict[tag] = 1 if tag not in tag_dict.keys() else tag_dict[tag] + 1
            tag = first + " " + tag
            tag_dict[tag] = 1 if tag not in tag_dict.keys() else tag_dict[tag] + 1
            # restore tag to uni
            tag = tag_store
            if word not in word_dict.keys():
                word_dict[word] = {}
            word_dict[word][tag] = 1 if tag not in word_dict[word].keys() \
                else word_dict[word][tag] + 1

            patterns = [key for key, pattern in Utils.regular_expressions.items() if pattern.match(word)]
            if patterns:
                for pat in patterns:
                    if pat not in word_dict.keys():
                        word_dict[pat] = {}
                    word_dict[pat][tag] = 1 if tag not in word_dict[pat].keys() else word_dict[pat][tag] + 1

            first = second
            second = third.rsplit('/', 1)[1]

    unk_list = [[word, tag, count] for word, tags in word_dict.items()
                for tag, count in tags.items() if count < 3 and not word.startswith('^')]
    for word, tag, value in unk_list:
        if tag in word_dict["*UNK*"].keys():
            word_dict["*UNK*"][tag] += value
        else:
            word_dict["*UNK*"][tag] = value

    with open(q_output, "w") as f:
        for key, value in tag_dict.items():
            f.write(f"{key}\t{value}\n")
    with open(e_output, "w") as f:
        for word, tags in word_dict.items():
            for tag, count in tags.items():
                key = word + " " + tag
                f.write(f"{key}\t{count}\n")

    return word_dict, tag_dict


if __name__ == "__main__":
    input_f = argv[1]
    q_output = argv[2]
    e_output = argv[3]
    parse_file(input_f, e_output, q_output)
