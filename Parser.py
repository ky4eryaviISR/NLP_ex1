from sys import argv
import re


regular_expressions ={'^ness NN': re.compile('\w+ness$'),
                      '^ise VB': re.compile("\w+ise$"),
                      '^ish JJ': re.compile('\w+ish$'),
                      '^ly RB': re.compile('\w+ly$'),
                      '^al JJ': re.compile('\w+al$'),
                      '^fy VB': re.compile('\w+fy$'),
                      '^able JJ': re.compile('\w+able$'),
                      '^ance NN': re.compile("\w+ance$"),
                      '^er NN': re.compile("\w+er$")
                      }

def getQ(t1, t2, t3):
    q_list = []
    total = 0
    lam1 = lam2 = lam3 = 0.33
    with open('q.mle') as f:
        for line in f:
            q_list.append(line.split())
            total += int(line.split()[-1])
    print("x")
    dict = {" ".join(item[:-1:1]): item[-1] for item in q_list}
    trigram_count = int(dict[f"{t1} {t2} {t3}"])
    bi_count = int(dict[f"{t1} {t2}"])
    uni_count = int(dict[t1])
    return lam1 * trigram_count / bi_count + lam2 * bi_count / uni_count + lam3 * uni_count / total


def parse_file(input_file, e_output, q_output):
    lines = [line.split() for line in open(input_file)]
    word_dict = {}
    for line in lines:
        for word, tag in [word.rsplit('/', 1) for word in line]:
            key = f"{word} {tag}"
            word_dict[key] = 1 if key not in word_dict.keys() else word_dict[key] + 1
            pat = [key for key, pattern in regular_expressions.items() if pattern.match(word)]
            if pat != []:
                pat = pat[0]
                word_dict[pat] = 1 if pat not in word_dict.keys() else word_dict[pat] + 1

    unk_dict = [[key.split()[-1], value] for key, value in word_dict.items() if value < 10]

    for key, value in unk_dict:
        if "*UNK* "+key in word_dict:
            word_dict["*UNK* "+key] += value
        else:
            word_dict["*UNK* "+key] = value


    with open(e_output, "w") as f:
        for key, value in word_dict.items():
            f.write(f"{key}\t{value}\n")

    tag_dict = {}
    for line in lines:
        for first, second, third in zip(line[::1], line[1::1], line[2::1]):
            key = first.rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
            key += " " + second.rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
            key += " " + third.rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1

        key = line[-1].rsplit('/', 1)[1]
        tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
        if len(line) > 1:
            key = line[-2].rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1
        if len(line) > 2:
            key += " " + line[-1].rsplit('/', 1)[1]
            tag_dict[key] = 1 if key not in tag_dict.keys() else tag_dict[key] + 1

    with open(q_output, "w") as f:
        for key, value in tag_dict.items():
            f.write(f"{key}\t{value}\n")



if __name__ == "__main__":
    input_f = argv[1]
    e_output = argv[2]
    q_output = argv[3]
    parse_file(input_f, e_output, q_output)
#    getQ("NNP", "POS", "NNP")