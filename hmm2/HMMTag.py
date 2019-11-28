from sys import argv

from Utils import Utils
from calculate_accuracy import calculate_accuracy
from hmm2.GreedyTag import greedyStart


def viterbi_start():
    print("X")
    lst = []

    for i in range(len(sentences)):
        for j in range(2,len(sentences[i])):
            for tag in utils.labels:
                # for prev
                probability = utils.getE(sentences[i][j][1], tag) * utils.getQ("START", "START", tag)


def V(i, t, prev, prev_prev):
    if i < 2:
        return 1
    word = sentences[0][i][1]
    return V(i-1,prev_prev)*utils.getE(word, t)*utils.getQ(t, prev, prev_prev)




if __name__ == "__main__":
    input_f = argv[1]
    e_output = argv[2]
    q_output = argv[3]
    output_f = argv[4]
    gamma = [0.15533849, 0.02582799, 0.81883352]
    utils = Utils(gamma)
    utils.load_q_and_load_e(q_output, e_output)
    sentences = utils.load_input_file(input_f)
    result = viterbi_start()
    utils.print_to_file(result, output_f)
    calculate_accuracy()