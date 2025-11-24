# Reset conda configuration
echo "Running conda configuration script! If the conda env is not present launch this script with -c"
/work/amazon_project/miniconda3/bin/conda init
source ~/.bashrc
cd /work/amazon_project
conda --version


#Installation of dot-env
#conda install python-dotenv

# Create conda environment
if [[ "$1" == "-c" ]]; then
    env_name="environment_new.yml"
    echo "You have passed $1 as params! Creating new environment from $env_name"
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main  #Accept condition of conda repo
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r     #Accept condition of conda repo
    conda env create -n NLP -f $env_name --yes
fi

# Activate conda environment
conda activate NLP
python --version

# Python script
pgm="experiment2.py"
echo "Running python training script $pgm"
#python $pgm
