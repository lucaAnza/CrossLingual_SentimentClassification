# Experiment1

## How to setup a run

1. Install conda [Conda official webpage](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
   - [Linux installer](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer)
2. Create a conda environment `conda env create -n NLP -f environment.yml`
   - If you add some package export again `conda env export > environment.yml`

## Usefull information

1. If you have problem with GPU-NVIDIA (device is not available to this):
   -   `sudo fuser -v /dev/nvidia*`
   -   `kill -9 <PID>` of the processes that are not about graphic server (such as Xorg)

