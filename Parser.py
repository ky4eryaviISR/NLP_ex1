from sys import argv


def getQ(t1,t2,t3):
    with open()


def parse_file(input_file,e_output,q_output):
    lines = [line.split() for line in open(input_file)]
    tag_dictionary = {}
    word_dictionary = {}
    for line in lines:
        bi_gram = tri_gram = None
        for word, tag in [word.rsplit('/',1) for word in line]:
            word_dictionary[word] = 1 if word not in word_dictionary else word_dictionary[word] + 1
            tag_dictionary[tag] = 1 if tag not in tag_dictionary else tag_dictionary[tag] + 1
            if bi_gram:
                key = tag + ' ' + bi_gram
                tag_dictionary[key] = 1 if key not in tag_dictionary else tag_dictionary[key] + 1
            if tri_gram:
                key = tag + ' ' + bi_gram + ' ' + tri_gram
                tag_dictionary[key] = 1 if key not in tag_dictionary else tag_dictionary[key] + 1
            tri_gram = bi_gram
            bi_gram = tag
    with open(q_output, "w") as f:
        for key, value in tag_dictionary.items():
            f.write(f"{key}\t{value}\n")
    with open(e_output, "w") as f:
        for key, value in word_dictionary.items():
            f.write(f"{key}\t{value}\n")



if __name__=="__main__":
    input_f = argv[1]
    e_output = argv[2]
    q_output = argv[3]
    parse_file(input_f, e_output, q_output)
