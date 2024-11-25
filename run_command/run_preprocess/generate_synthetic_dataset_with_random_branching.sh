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
--save-to '../data/synthetic_random/nozero/'

python synthetic_dataset.py \
--final-level 4 \
--max-branching 10 \
--seed 42 \
--zero-probability 0.5 \
--max-flow 1000 \
--random-branching \
--scale 3 \
--save-to '../data/synthetic_random/dense/'

python synthetic_dataset.py \
--final-level 4 \
--max-branching 10 \
--seed 42 \
--zero-probability 0.99 \
--max-flow 1000 \
--random-branching \
--scale 3 \
--save-to '../data/synthetic_random/sparse/'

