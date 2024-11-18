# go back two folder
cd ..
# enter in the folder preprocess_data
cd experiments || exit
# run the python program synthetic_dataset.py

python Italy_different_norms.py --delta 1e-8 --show-tqdm --epsilons 10 --num-experiments 1 --final-level 2