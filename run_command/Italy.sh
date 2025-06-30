# go back two folder
cd ..
# enter in the folder preprocess_data
cd experiments || exit
# run the python program synthetic_dataset.py

python Italy_experiments.py --delta 1e-8 --show-tqdm --epsilons 0.1,1,10 --num-experiments 10 --final-level 6 \
--file-path "../data/Italy" \
--save-path "../results/Italy"
