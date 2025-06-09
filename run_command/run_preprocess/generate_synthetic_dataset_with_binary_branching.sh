# go back two folder
cd ../..
# enter in the folder preprocess_data
cd preprocess_data || exit
# run the python program synthetic_dataset.py

python synthetic_dataset.py \
--final-level 8 \
--max-branching 2 \
--seed 42 \
--zero-probability 0 \
--max-flow 1000 \
--scale 3 \
--save-to '../data/synthetic_binary/complete/'

python synthetic_dataset.py \
--final-level 8 \
--max-branching 2 \
--seed 42 \
--zero-probability 0.5 \
--max-flow 1000 \
--scale 3 \
--save-to '../data/synthetic_binary/dense/'

python synthetic_dataset.py \
--final-level 8 \
--max-branching 2 \
--seed 42 \
--zero-probability 0.99 \
--max-flow 1000 \
--scale 3 \
--save-to '../data/synthetic_binary/sparse/'
