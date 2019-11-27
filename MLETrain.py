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

    def __init__(self, input, e_mle, q_mle,gamma=[1/3,1/3,1/3]):
        self.e_dict, self.q_dict = self.parse_file(input, e_mle, q_mle)
        self.gamma1, self.gamma2, self.gamma3 = gamma
        print("x")
        self.labels = set()
    
    def getE(self, w, t):
        if w not in self.e_dict or t not in self.e_dict[w]:
            pat = [key for key, pattern in self.regular_expressions.items() if pattern.match(w)]
            if pat:
                w = pat
            else:
                w = '*UNK*'
        return self.e_dict[w][t] / sum(self.e_dict[w].values())



    def getQ(self, t1, t2, t3):
        p_t3 = self.q_dict[t3]/self.totalWords
        p_t3_t2 = self.q_dict[t2+' '+t3]/self.q_dict[t2]
        p_t3_t2_t1 = self.q_dict[t1+' '+t2+' '+t3]/self.q_dict[t1+' '+t2]
        return self.gamma1*p_t3+self.gamma2*p_t3_t2+self.gamma3*p_t3_t2_t1



    def parse_file(self, input_file, e_output, q_output):
        lines = [line.split() for line in open(input_file)]
        self.totalWords = 0
        word_dict = {"*UNK*": {}}
        tag_dict = {}
        for line in lines:
            first = second = "START"
            for third in line:
                self.totalWords += 1
                word, tag = third.rsplit('/', 1)
                tag_store = tag
                self.labels.add(tag)
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

                pat = [key for key, pattern in self.regular_expressions.items() if pattern.match(word)]
                if pat:
                    pat = pat[0]
                    if pat not in word_dict.keys():
                        word_dict[pat] = {}
                    word_dict[pat][tag] = 1 if tag not in word_dict[pat].keys() else word_dict[pat][tag] + 1

                first = second
                second = third.rsplit('/', 1)[1]

        unk_list = [[word, tag, count] for word, tags in word_dict.items()
                    for tag, count in tags.items() if count < 5 and not word.startswith('^')]
        for word, tag, value in unk_list:
            word_dict[word].pop(tag)
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
    e_output = argv[2]
    q_output = argv[3]
    MLETrain(input_f, e_output, q_output)
#    getQ("NNP", "POS", "NNP")