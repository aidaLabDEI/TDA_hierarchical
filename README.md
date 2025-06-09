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
