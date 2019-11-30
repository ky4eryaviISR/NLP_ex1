import copy
from datetime import datetime
import math
from functools import reduce
from sys import argv
import numpy as np

from Utils import Utils
from calculate_accuracy import calculate_accuracy
import multiprocessing as mp


start = datetime.now()
input_f = argv[1]
e_output = argv[2]
q_output = argv[3]
output_f = argv[4]
gamma = [0.01, 0.09, 0.9]
utils = Utils()
utils.load_q_and_load_e(q_output, e_output)
prev_tag_list = utils.get_possible_prev_tag()
cross_tags = utils.get_cross_tags()
sentences = utils.load_input_file(input_f)

def viterbi_start(start, stop, sentences):

    for i in range(start, stop):
        viterbi = [copy.deepcopy(cross_tags), copy.deepcopy(cross_tags)]
        viterbi[1]["START"]["START"] = 0

        tags = []
        for j in range(2, len(sentences[i])):
            word = sentences[i][j][1]
            word_score = copy.deepcopy(cross_tags)
            best_tags = copy.deepcopy(cross_tags)
            if word not in utils.e_dict.keys():
                word = utils.change_word(word)
            for t_third in utils.e_dict[word].keys():
                emission = utils.getE(word, t_third)
                if emission == -np.inf:
                    continue
                for t_second in prev_tag_list[t_third]:
                    best_score = -np.inf
                    first_tag = ""
                    for t_first in prev_tag_list[t_second]:
                        new_score = calculate_score(emission,
                                                    t_third,
                                                    t_second,
                                                    t_first,
                                                    viterbi[j-1][t_second][t_first])
                        if new_score > best_score:
                            best_score = new_score
                            word_score[t_third][t_second] = new_score
                            best_tags[t_third][t_second] = t_first

            #viterbi.pop(0)
            viterbi.append(word_score)
            tags.append(best_tags)

        cur_score = -np.inf
        last_tag = prev_tag = ""
        for prev, prev__prev_keys in viterbi[-1].items():
            for prev_prev, score in prev__prev_keys.items():
                if cur_score < score:
                    cur_score = score
                    prev_tag = prev_prev
                    last_tag = prev

        sentences[i][-1][0] = last_tag
        sentences[i][-2][0] = prev_tag

        for index in range(len(tags)-1, 1, -1):
            sentences[i][index][0] = tags[index][last_tag][prev_tag]
            last_tag = prev_tag
            prev_tag = sentences[i][index][0]
        sentences[i] = sentences[i][2:]
        #print(i)
    return sentences[start: stop]


def calculate_score(emission, t, prev, prev_prev, viterbi):
    transition = utils.getQ(prev_prev, prev, t)
    if transition == 0:
        return -np.inf
    return np.log(transition) + np.log(emission) + viterbi


if __name__ == "__main__":
    num_sentences = len(sentences)
    steps = math.ceil((num_sentences / mp.cpu_count()))
    args = []
    for i in range(0, num_sentences, steps):
        args.append([i, min(i+steps, num_sentences), sentences])
    for i in range(1):
        print("START NEW")
        #gamma = np.random.dirichlet(np.ones(3), size=1)[0]
        #print(gamma)
        #utils.gamma1, utils.gamma2, utils.gamma3 = gamma

        pool = mp.Pool(mp.cpu_count())
        results = pool.starmap(viterbi_start, args)
        results = reduce(lambda x, y: x + y, results)
        utils.print_to_file(results, output_f)
        #print(datetime.now() - start)
        calculate_accuracy('viterbi_hmm_output.txt', '../data/ass1-tagger-dev')
