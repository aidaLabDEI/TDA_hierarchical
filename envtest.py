## Test the environment
"""
jupyter==1.1.1
pandas==2.2.3
opendp==0.11.1
tqdm==4.66.5
cvxpy==1.5.3
cvxopt==1.3.2
matplotlib==3.9.2
numba==0.60.0
"""
import opendp
import pandas as pd
import tqdm
import cvxpy as cp
import cvxopt
import matplotlib
import numba
import importlib.metadata


# check the versions
def check_versions():
    # print("OpenDP version:", dp.__version__)
    pd_version = pd.__version__
    if pd_version != '2.2.3':
        print("Current pandas version:", pd_version, "Recommended version: 2.2.3")
    tqdm_version = tqdm.__version__
    if tqdm_version != '4.66.5':
        print("Current tqdm version:", tqdm_version, "Recommended version: 4.66.5")
    cvxopt_version = cvxopt.__version__
    if cvxopt_version != '1.3.2':
        print("Current cvxopt version:", cvxopt_version, "Recommended version: 1.3.2")
    cp_version = cp.__version__
    if cp_version != '1.5.3':
        print("Current cvxpy version:", cp_version, "Recommended version: 1.5.3")
    numba_version = numba.__version__
    if numba_version != '0.60.0':
        print("Current numba version:", numba_version, "Recommended version: 0.60.0")
    opendp_version = importlib.metadata.version("opendp")
    if opendp_version != '0.11.1':
        print("Current OpenDP version:", opendp_version, "Recommended version: 0.11.1")
    matplotlib_version = importlib.metadata.version("matplotlib")
    if matplotlib_version != '3.9.2':
        print("Current matplotlib version:", matplotlib_version, "Recommended version: 3.9.2")


if __name__ == "__main__":
    check_versions()
    print("Environment is set up correctly.")
