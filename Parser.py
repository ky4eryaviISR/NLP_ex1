from sys import argv


class Tree_gram(object):
    def __init__(self,args):
        self.trigram = [args[0], args[1], args[2]]
        self.count = int(args[3])

    def __eq__(self, other):
        return self.trigram == other.trigram

    def approximate(self,a,b):
        if a in self.trigram and b in self.trigram:
            return True
        return False




# def getQ(t1,t2,t3):
    # q_list = []
    # with open('q.mle') as f:
    #     for line in f:
    #         q_list.append(Tree_gram(line.split()))
    # count_t1_t2_t3 = count_t1_t2 = 0
    # for tree_gram in q_list:
    #     if tree_gram == Tree_gram([t1, t2, t3, 0]):
    #         count_t1_t2_t3 += tree_gram.count
    #         count_t1_t2 += tree_gram.count
    #     elif tree_gram.approximate(t1, t2):
    #         count_t1_t2 += tree_gram.count
    #



def parse_file(input_file,e_output,q_output):
    lines = [line.split() for line in open(input_file)]
    word_dict = {}
    for line in lines:
        for word, tag in [word.rsplit('/',1) for word in line]:
            key = f"{word} {tag}"
            word_dict[key] = 1 if key not in word_dict.keys() else word_dict[key] + 1

    with open(e_output, "w") as f:
        for key, value in word_dict.items():
            f.write(f"{key}\t{value}\n")

    tag_dict = {}
    for line in lines:
        for first,second,third in zip(line[::1], line[1::1], line[2::1]):
            key = first.rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
            key += " " + second.rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
            key += " " + third.rsplit('/',1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1

        key = line[-1].rsplit('/', 1)[1]
        tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
        if len(line) > 1:
            key = line[-2].rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
        if len(line) > 2:
            key += " " + line[-1].rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1

    print("x")
    with open(q_output, "w") as f:
        for key, value in tag_dict.items():
            f.write(f"{key}\t{value}\n")

    # for line in lines:
    #     bi_gram = tri_gram = None
    #     for word, tag in [word.rsplit('/',1) for word in line]:
    #         word_dictionary[word] = 1 if word not in word_dictionary else word_dictionary[word] + 1
    #         tag_dictionary[tag] = 1 if tag not in tag_dictionary else tag_dictionary[tag] + 1
    #         if bi_gram:
    #             key = tag + ' ' + bi_gram
    #             tag_dictionary[key] = 1 if key not in tag_dictionary else tag_dictionary[key] + 1
    #         if tri_gram:
    #             key = tag + ' ' + bi_gram + ' ' + tri_gram
    #             tag_dictionary[key] = 1 if key not in tag_dictionary else tag_dictionary[key] + 1
    #         tri_gram = bi_gram
    #         bi_gram = tag
    # with open(q_output, "w") as f:
    #     for key, value in tag_dictionary.items():
    #         f.write(f"{key}\t{value}\n")
    # with open(e_output, "w") as f:
    #     for key, value in word_dictionary.items():
    #         f.write(f"{key}\t{value}\n")



if __name__=="__main__":
    input_f = argv[1]
    e_output = argv[2]
    q_output = argv[3]
    parse_file(input_f, e_output, q_output)
    # getQ("NNP", "POS", "NNP")
