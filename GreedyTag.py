from datetime import datetime
from sys import argv
import operator
from Utils import Utils
from calculate_accuracy import calculate_accuracy
import numpy as np

def get_best_tag(a, b, c):
    probabilities = {label: utils.getE(c[1], label) * utils.getQ(a[0], b[0], label)
                     for label in utils.labels}
    return max(probabilities.items(), key=operator.itemgetter(1))[0]


def greedyStart(sentences):
    result = []
    for i in range(len(sentences)):
        result.append([])
        for j in range(2, len(sentences[i])):
            tag = get_best_tag(sentences[i][j-2], sentences[i][j-1], sentences[i][j])
            sentences[i][j][0] = tag
            result[i].append([tag, sentences[i][j][1]])

    return result


if __name__ == "__main__":
    print(datetime.now())
    input_f = argv[1]
    q_output = argv[2]
    e_output = argv[3]
    output_f = argv[4]
    gamma = [0.15533849, 0.02582799, 0.81883352]
    utils = Utils(gamma)
    utils.load_q_and_load_e(q_output, e_output)
    sentences = utils.load_input_file(input_f)
    result = greedyStart(sentences)
    utils.print_to_file(result, output_f)
    print(datetime.now())
