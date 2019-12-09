# NLP_1
HomeWork in NLP Bar Ilan

## Creating Transmission and Emission file
python MLETrain.py data/ass1-tagger-train q.mle e.mle

## HMM using Greedy algorithm
python GreedyTag.py data/ass1-tagger-dev-input q.mle e.mle data/greedy_hmm_output.txt

## HMM using Viterbi algorithm
python HMMTag.py data/ass1-tagger-dev-input q.mle e.mle data/viterbi_hmm_output.txt

## Extract Features
python ExtractFeatures.py data/ass1-tagger-train features_file

## Training Logistic Regression
python TrainSolver.py features_file model_file

## MEMM using Greedy Algorithm
python GreedyMaxEntTag.py data/ass1-tagger-dev-input model_file feature_map_file memm-greedy-predictions.txt

## MEMM using Viterbi Algorithm
python MEMMTag.py data/ass1-tagger-dev-input model_file feature_map_file memm-viterbi-predictions.txt

