# go back two folder
cd ..
# enter in the folder preprocess_data
cd experiments || exit
# run the python program synthetic_dataset.py

python Italy_different_norms.py --delta 1e-8 --show-tqdm --epsilons 1,10 --num-experiments 5 --final-level 6 \
--save-path "../results/Italy/different_norms"