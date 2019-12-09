

def calculate_accuracy(actual_txt,predicted_txt):
    predicted = []
    actual = []
    act_word =[]
    predicted_word = []
    with open(actual_txt) as f:
        for line in f:
            actual.append([word.rsplit('/', 1)[1] for word in line.split()])
            act_word.append([word.rsplit('/', 1)[0] for word in line.split()])
    with open(predicted_txt) as f:
        for line in f:
            predicted.append([word.rsplit('/', 1)[1] for word in line.split()])
            predicted_word.append([word.rsplit('/', 1)[0] for word in line.split()])
    good = count = 0
    for tag_actual, tag_pred, a_w,p_w in zip(actual, predicted, act_word, predicted_word):
        for i in range(len(tag_actual)):
            if tag_pred[i] == tag_actual[i]:
                good += 1
            # else:
            #     print(tag_actual[i],a_w[i],tag_pred[i],p_w[i])
            count += 1
    return (good/count)




if __name__ == '__main__':
    print("HMM greedy",calculate_accuracy('data/ass1-tagger-dev', 'data/greedy_hmm_output.txt'))
    print("HMM viterbi",calculate_accuracy('data/ass1-tagger-dev', 'data/viterbi_hmm_output.txt'))
    print("MEMM greedy",calculate_accuracy('data/ass1-tagger-dev', 'data/memm-greedy-predictions.txt'))
    print("MEMM viterbi",calculate_accuracy('data/ass1-tagger-dev', 'data/memm-viterbi-predictions.txt'))
    print("NER HMM greedy",calculate_accuracy('ner/dev','ner/greedy_hmm_output.txt'))
    print("NER HMM viterbi",calculate_accuracy('ner/dev','ner/viterbi_hmm_output.txt'))
    print("NER MEMM greedy",calculate_accuracy('ner/dev','ner/memm-greedy-predictions.txt'))
    print("NER MEMM viterbi",calculate_accuracy('ner/dev','ner/memm-viterbi-predictions.txt'))
