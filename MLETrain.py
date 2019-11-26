from sys import argv
import re


class MLETrain(object):

    regular_expressions ={'^ness ': re.compile('\w+ness$'),
                          '^ise ': re.compile("\w+ise$"),
                          '^ish ': re.compile('\w+ish$'),
                          '^ly ': re.compile('\w+ly$'),
                          '^al ': re.compile('\w+al$'),
                          '^fy ': re.compile('\w+fy$'),
                          '^able ': re.compile('\w+able$'),
                          '^ance ': re.compile("\w+ance$"),
                          '^er ': re.compile("\w+er$"),
                          '^ed ': re.compile("\w+ed$"),
                          '^ing ': re.compile("\w+ing$"),
                          '^Aa ': re.compile("[A-Z][a-z]+$"),
                          '^0-9.0-9 ': re.compile("^\d+\.?\d+$"),
                          '^AA': re.compile("^[A-Z][A-Z]+$")
                          }

    def __init__(self, input, e_mle, q_mle):
        self.e_dict, self.q_dict = self.parse_file(input, e_mle, q_mle)
        print("x")

    def getE(self, w, t):


    def getQ(self, t1, t2, t3):


    def parse_file(self, input_file, e_output, q_output):
        lines = [line.split() for line in open(input_file)]

        word_dict = {}
        tag_dict = {}
        for line in lines:
            first = second = "START"
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
                if tag not in word_dict:
                    word_dict[tag] = {}
                word_dict[tag][word] = 1 if word not in word_dict[tag].keys() \
                    else word_dict[tag][word] + 1

                pat = [key for key, pattern in self.regular_expressions.items() if pattern.match(word)]
                if pat:
                    pat = pat[0]
                    word_dict[tag][pat] = 1 if pat not in word_dict[tag].keys() else word_dict[tag][pat] + 1

                first = second
                second = third.rsplit('/', 1)[1]

        unk_list = [[word, tag, count] for tag, words in word_dict.items()
                    for word, count in words.items() if count < 5 and not word.startswith('^')]

        for word, tag, value in unk_list:
            word_dict[tag].pop(word)
            if "*UNK* " in word_dict[tag].keys():
                word_dict[tag]["*UNK*"] += value
            else:
                word_dict[tag]["*UNK*"] = value

        with open(q_output, "w") as f:
            for key, value in tag_dict.items():
                f.write(f"{key}\t{value}\n")
        with open(e_output, "w") as f:
            for tag, words in word_dict.items():
                for word, count in words.items():
                    key = word + " " + tag
                    f.write(f"{key}\t{count}\n")

        return word_dict, tag_dict


if __name__ == "__main__":
    input_f = argv[1]
    e_output = argv[2]
    q_output = argv[3]
    MLETrain(input_f, e_output, q_output)
#    getQ("NNP", "POS", "NNP")