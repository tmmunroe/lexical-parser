predictors=(smurf freq lesk word2vec bert bertbest ensemble)

for predictor in ${predictors[@]}
do
    filename="$predictor.predict"
    echo "Evaluating $predictor predictor and writing to $filename"
    python lexsub.py $predictor > $filename
    perl score.pl $filename gold.trial
    echo "--------------------------------------------------"
    echo "--------------------------------------------------"
done