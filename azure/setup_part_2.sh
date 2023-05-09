# Setup

MYHOME=/home/azureuser
CONDA=$MYHOME/miniconda3/bin/conda
CONDA_ENV=mla

# Create conda env

$CONDA create -n $CONDA_ENV python=3.8 -y

# Install mlagility

cd $MYHOME
git clone https://github.com/groq/mlagility.git
$CONDA run -n $CONDA_ENV pip install -e mlagility
$CONDA run -n $CONDA_ENV pip install -r mlagility/models/requirements.txt

# selftest

$CONDA run -n $CONDA_ENV benchit $MYHOME/mlagility/models/selftest/linear.py