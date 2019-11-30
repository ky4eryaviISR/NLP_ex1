# NLP_1
HomeWork in NLP Bar Ilan

## Creating Transmission and Emission file
python MLETrain.py data/ass1-tagger-train data/e.mle data/q.mle

## HMM using Greedy algorithm
python GreedyTag.py data/ass1-tagger-dev-input data/e.mle data/q.mle data/greedy_hmm_output

## HMM using Viterbi algorithm
python HMMTag.py data/ass1-tagger-dev-input data/e.mle data/q.mle data/viterbi_hmm_output.txt

## Extract Features
python ExtractFeatures.py data/ass1-tagger-train data/features_file

