# Differentially Private Release of Hierarchical Origin/Destination Data with a TopDown Approach

This repository contains the code for the paper "Differentially Private Release of Hierarchical Origin/Destination Data with a TopDown Approach"

To run the code, you need to install a conda environment using the environment.yml file

```
conda env create -f environment.yml
```

Then, activate the environment

```
conda activate top-down
```
## Data Pre-Processing
### Generate the synthetic datasets
To generate the **synthetic** dataset run the shell files into `/run_command/run_preprocess/` folder. Inside you will
find two shell files, one for the binary tree, the other for the random tree. You can change the parameters of the
synthetic dataset in the shell files, like the sparsity, the number of levels, the seed for the randomizer.
```bash
cd run_command/run_preprocess

chmod +x generate_synthetic_dataset_with_binary_branching.sh
chmod +x generate_synthetic_dataset_with_random_branching.sh

./generate_synthetic_dataset_with_binary_branching.sh
./generate_synthetic_dataset_with_random_branching.sh

```

### Generate the real dataset
It is necessary to download the dataset from ISTAT website. 

https://www.istat.it/storage/cartografia/matrici_pendolarismo/matrici-pendolarismo-sezione-censimento-2011.zip

This files needs to be inserted into the `/preprocess_data` directory. 
Then, it is sufficient to run the python script 

```bash
cd preprocess_data
python preprocess_ISTAT_data.py
```

This generates a data folder containing the datasets.

## Experiments
The experiment on the Italian dataset can be run using the shell file `/run_command/Italy.sh`

```bash
cd run_command
chmod +x Italy.sh
./Italy.sh
```
The experiments on the synthetic dataset can be run using the shell file `/run_command/all_synthetic.sh`

```bash
cd run_command
chmod +x all_synthetic.sh
./all_synthetic.sh
```

The experiments will generate new `results.pickle` files in the `/results` folder. The results presented in the paper
are already present in the folder as `results_1.pickle`. To plot the results, you can use the jupyter notebook in the
`/notebooks` folder.

### Note
1. The experiments take a long time to run. The Italy dataset takes about 2 hours per experiments while the synthetic
datasets take about 1 hour. By default, 10 indipendent experiments are run for each dataset, so the total time surpasses a day.
I recommend to run just for a couple of experiments. It is sufficient to change th shell files.
Example on Italy dataset
```
python Italy_experiments.py --delta 1e-8 --show-tqdm --epsilons 0.1,1,10 --num-experiments 10 --final-level 6 \
--file-path "../data/Italy" \
--save-path "../results/Italy"
```
Change to
```
python Italy_experiments.py --delta 1e-8 --show-tqdm --epsilons 0.1,1,10 --num-experiments 1 --final-level 6 \
--file-path "../data/Italy" \
--save-path "../results/Italy"
```

2.  Even though our paper investigates theoretically how to generalize the algorithm for different sensitivities, our implementation works only for the case where each user contributes to a single trip, and the neighboring relation is for bounded differential privacy. This is the case investigated in our experimental section.

