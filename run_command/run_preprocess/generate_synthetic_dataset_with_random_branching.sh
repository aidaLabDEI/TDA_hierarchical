# go back two folder
cd ../..
# enter in the folder preprocess_data
cd preprocess_data || exit
# run the python program synthetic_dataset.py

python synthetic_dataset.py \
--final-level 4 \
--max-branching 10 \
--seed 42 \
--zero-probability 0 \
--max-flow 1000 \
--random-branching \
--scale 3 \
--save-to '../data/synthetic_random_branching_nozero/'

python synthetic_dataset.py \
--final-level 4 \
--max-branching 10 \
--seed 42 \
--zero-probability 0.5 \
--max-flow 1000 \
--random-branching \
--scale 3 \
--save-to '../data/synthetic_random_branching_dense/'

python synthetic_dataset.py \
--final-level 4 \
--max-branching 10 \
--seed 42 \
--zero-probability 0.99 \
--max-flow 1000 \
--random-branching \
--scale 3 \
--save-to '../data/synthetic_random_branching_sparse/'

