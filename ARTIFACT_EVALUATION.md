# Artifact Appendix

Paper title: **Differentially Private Release of Hierarchical Origin/Destination
Data with a TopDown Approach**

Artifacts HotCRP Id: **14**

Requested Badge: **Available**, **Functional**, **Reproduced**

## Description

My artifacts consits in python code, a conda enviroment, and a real dataset. The
`README.md` file in the root directory contains all the information to download the real dataset
(at submission time the link is working, if it will not in the future is sufficient to reach me out), to
install and set up the conda environment, and to run the experiments. To simplify reproducibility,
all the experements can be run using the shell files in the `run_command` folder.
The experiments produce .pickle files that can be used to obtain the plots using
the provided jupyter notebook in the `notebooks` folder.

### Security/Privacy Issues and Ethical Concerns (All badges)

None

## Basic Requirements (Only for Functional and Reproduced badges)

A lapton with medium performance is sufficient to run the experiments.

### Hardware Requirements

Nothing specific

### Software Requirements

All the software requirements are listed in the `environment.yml` file. The libraries used are all Open Source.

### Estimated Time and Storage Consumption

To get synthetic datasets, and pre-process the Italy dataset, take a few minutes. 
Italy dataset 10 experiments take about 20 hours to run, while the synthetic datasets take about 10 minutes to run all the

## Environment

To run the code, you need to install a conda environment using the environment.yml file

```bash
conda env create -f environment.yml
```

Then, activate the environment

```bash
conda activate top-down
```

### Accessibility (All badges)

ADD commit id

### Set up the environment (Only for Functional and Reproduced badges)

See Environment section above.

### Testing the Environment (Only for Functional and Reproduced badges)

After creating the conda enviroment and activating it, you can run the following command to test if everything is
working correctly:

```bash
python envtest.py
```

If everything is working correctly, you should see the message "Environment test passed" printed in the terminal.
Otherwise, you'll see the current version of the libraries.

## Artifact Evaluation (Only for Functional and Reproduced badges)

This section includes all the steps required to evaluate your artifact's functionality and validate your paper's key
results and claims.
Therefore, highlight your paper's main results and claims in the first subsection. And describe the experiments that
support your claims in the subsection after that.

### Main Results and Claims

InfTDA is as accurate as other optimization methods for different privacy budgets $\varepsilon = [0.1, 1, 10]$, it
reduces
effectivetly the number of false positives, and the implementation is faster.
This for both synthetic and real datasets. This is corrobareted by the plots that can be obtained by first running the
shell
files `Italy.sh` and `all_synthetic.sh` in the `run_command` folder, and then running the jupyter notebooks
`plots_Italy.ipynb` and
`synthetic_dataset.ipynb`.

#### Main Result 1: Utility

The utility experiments regards the maximum absolute error for each hierarchy level.

#### Main Result 2: False Positives

The false positives experiments regards the false discovery rate for each hierarchy level.

#### Main Result 3: Efficiency

The efficiency is the time taken to each algorithm. In the experiments were run using a remote machine with the
following
capabilities: Intel Xeon Processor W-2245 (8
cores, 3.9GHz), 128GB RAM, and Ubuntu 20.04.3. Thus, running on different machines may lead to different results.
Though, infTDA
is expected to be faster than the other algorithms.

### Experiments

- Download the real dataset from ISTAT website https://www.istat.it/storage/cartografia/matrici_pendolarismo/matrici-pendolarismo-sezione-censimento-2011.zip
and put it in the `preprocess_data` folder.
- Pre-process the real dataset by running the script 
```bash
python preprocess_data/preprocess_ISTAT_data.py
```
This generates a data folder containing the datasets.
- To generate the **synthetic** dataset, run the script
```bash
cd run_command/run_preprocess

chmod +x generate_synthetic_dataset_with_binary_branching.sh
chmod +x generate_synthetic_dataset_with_random_branching.sh

./generate_synthetic_dataset_with_binary_branching.sh
./generate_synthetic_dataset_with_random_branching.sh

```
- Run `Italy.sh` and `all_synthetic.sh` in the `run_command` folder to get results for the real and synthetic datasets,
  respectively. 

```bash
cd run_command
chmod +x Italy.sh
./Italy.sh
```
```bash
cd run_command
chmod +x all_synthetic.sh
./all_synthetic.sh
```

  This generates files `.pickle` in the results folder. In that folder are already present the results of my paper, they are
  labelled as `results_1.pickle`.
  Generating new data will increase the counter of the label to `results_2.pickle`.
  After that, run the jupyter notebooks `plots_Italy.ipynb` and `synthetic_dataset.ipynb` in the `notebooks` folder to
  obtain the plots. Be sure to
  change the initial `filename` variable to `results_2.pickle` or by default it will use the `results_1.pickle` file.
- For the Italy dataset, running a single experiment takes about 2 hours. This includes running all the benchmarks for three privacy budgets. Thus
in total it might take up to 20 hours to run all the 10 experiments. 
- You should obtain plots similar to the ones in the paper, even by running over different results. The same plots can
  be obtained by running the notebooks
  using the `results_1.pickle` file, which is already present in the `results` folder.

## Limitations (Only for Functional and Reproduced badges)

All the plots are reproducible.

## Notes on Reusability (Only for Functional and Reproduced badges)
The current implementation is not designed to be used as a library, but rather as a tool to run the experiments.



