from sys import argv
from  MLETrain import MLETrain

def greedyStart(trainer):
    lines = [line.split() for line in open(input_file)]
    for line in lines:
        first = second = "START"
        for third in line:
            word, tag = third.rsplit('/', 1)
            trainer.labels.add(tag)



if __name__ == "__main__":
    input_f = argv[1]
    e_output = argv[2]
    q_output = argv[3]
    greedyStart(MLETrain(input_f, e_output, q_output))
