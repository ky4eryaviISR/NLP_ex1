import re
import numpy as np
regular_expressions = {'^ness ': re.compile('\w+ness$'),
                       '^0-9.0-9 ': re.compile("^\d+\.?\,?\:?\-?\d+$"),
                       "^A-Z": re.compile("[A-Z]+$"),
                       "^ing": re.compile("\w+ing$"),
                       "^ed": re.compile("\w+ed$"),
                       "^-": re.compile("\w+-\w+$"),
                       "^'": re.compile("'\w+$"),
                       "^s": re.compile("\w+s$"),
                       "^ion": re.compile("\w+ion$"),
                       "^er": re.compile("\w+er$"),
                       "^ec": re.compile("\w+ec$"),
                       "^al": re.compile("\w+al$"),
                       '^Aa': re.compile('\w[A-Z]+[a-z]+$')}


class Utils(object):

    def __init__(self, gamma=(0.01, 0.09, 0.9)):
        self.e_dict = {}
        self.q_dict = {}
        self.e_dict_tag = {}
        self.gamma1, self.gamma2, self.gamma3 = gamma
        self.totalWords = 0
        self.labels = set()
        self.uni_gram = {}
        self.bi_gram = {}
        self.tri_gram = {}

    def load_input_file(self, input_file):
        data = []
        with open(input_file) as f:
            for line in f:
                data.append([["START", "*"], ["START", "*"]] + [['*', word] for word in line.split()])
        return data

    def print_to_file(self, result, output_file):
        with open(output_file, "w") as f:
            for sentence in result:
                output = ""
                for tag, word in sentence:
                    output += word + '/' + tag + ' '
                f.write(output + '\n')

    def load_q_and_load_e(self, q_file, e_file):
        with open(q_file) as f:
            q_dict = {line.rsplit('\t')[0]: int(line.rsplit('\t')[1]) for line in f}
            self.uni_gram = {key: value for key, value in q_dict.items() if len(key.split()) == 1}
            self.bi_gram = {key: value for key, value in q_dict.items() if len(key.split()) == 2}
            self.tri_gram = {key: value for key, value in q_dict.items() if len(key.split()) == 3}
        with open(e_file) as f:
            for line in f:
                word, tag, count = line.split()
                self.labels.add(tag)
                if word not in self.e_dict:
                    self.e_dict[word] = {}
                self.e_dict_tag[tag] = self.e_dict_tag.get(tag, 0) + int(count)
                self.e_dict[word][tag] = int(count)

        self.totalWords = sum(self.uni_gram.values())

    def getQ(self, t1, t2, t3):
        p_t3 = self.uni_gram[t3] / self.totalWords
        p_t3_t2 = self.bi_gram.get(t2 + ' ' + t3, 0) / self.uni_gram[t2]
        p_t3_t2_t1 = self.tri_gram.get(t1 + ' ' + t2 + ' ' + t3, 0) / self.bi_gram.get(t1 + ' ' + t2, 1)
        return self.gamma1 * p_t3 + self.gamma2 * p_t3_t2 + self.gamma3 * p_t3_t2_t1

    def change_word(self, w):
        pat = [key for key, pattern in regular_expressions.items() if pattern.match(w)]
        if pat:
            return pat[0].strip()
        return '*UNK*'

    def getE(self, w, t):
        if w not in self.e_dict:
            w = self.change_word(w)
        if t not in self.e_dict[w]:
            return -np.inf
        return self.e_dict[w].get(t, 0) / self.e_dict_tag[t]

    def get_possible_prev_tag(self):
        return {key: {value for value in self.uni_gram.keys()
                      if value + " " + key in self.bi_gram.keys()}
                for key in self.uni_gram.keys()}

    def get_cross_tags(self):
        return {key: {value: -np.inf for value in self.uni_gram.keys()}
                for key in self.uni_gram.keys()}
