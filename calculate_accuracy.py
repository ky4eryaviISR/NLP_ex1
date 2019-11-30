

def calculate_accuracy(predicted_txt,actual_txt):
    predicted = []
    actual = []
    with open(predicted_txt) as f:
        for line in f:
            actual.append([word.rsplit('/', 1)[1] for word in line.split()])
    with open(actual_txt) as f:
        for line in f:
            predicted.append([word.rsplit('/', 1)[1] for word in line.split()])
    good = count = 0
    for tag_actual, tag_pred in zip(actual, predicted):
        for i in range(len(tag_actual)):
            if tag_pred[i] == tag_actual[i]:
                good += 1
            count += 1
    print(good/count)




if __name__ == '__main__':
    calculate_accuracy()