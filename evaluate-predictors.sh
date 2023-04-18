predictors=(smurf freq lesk word2vec bert)

for predictor in ${predictors[@]}
do
    echo "Evaluating $predictor predictor"
    filename=$predictor.predict
    python lexsub.py $predictor > filename
    perl score.pl filename gold.trial
    echo "--------------------------------------------------"
    echo "--------------------------------------------------"
done