# Setup

MYUSER=azureuser
MYHOME=/home/$MYUSER

# Install NFS

# sudo apt-get -y update
# sudo apt-get -y install nfs-common
# sudo mkdir -p /mount/mlagilitystorage/mla-share

# Install Docker

sudo apt-get update
sudo apt-get install \
    ca-certificates \
    curl \
    gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get -y update

sudo apt-get -y install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

sudo usermod -aG docker $MYUSER

sudo service docker start

# bashrc

echo 'export MLAGILITY_DEBUG=True' >> $MYHOME/.bashrc
echo 'export ML_CACHE=/mount/mlagilitystorage/mla-share/ml-cache' >> $MYHOME/.bashrc
echo 'export HF_DATASETS_CACHE="$ML_CACHE""/huggingface"' >> $MYHOME/.bashrc
echo 'export TRANSFORMERS_CACHE="$ML_CACHE""/huggingface"' >> $MYHOME/.bashrc
echo 'export HF_HOME="$ML_CACHE""/huggingface"' >> $MYHOME/.bashrc
echo 'export XDG_CACHE_HOME="$ML_CACHE""/huggingface"' >> $MYHOME/.bashrc
echo 'export TORCH_HOME="$ML_CACHE""/torch-hub"' >> $MYHOME/.bashrc
echo 'export TORCH_HUB="$ML_CACHE""/torch-hub"' >> $MYHOME/.bashrc
# echo 'sudo mount -t nfs mlagilitystorage.file.core.windows.net:/mlagilitystorage/mla-share /mount/mlagilitystorage/mla-share -o vers=4,minorversion=1,sec=sys' >> $MYHOME/.bashrc

# conda

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $MYHOME/miniconda3
$MYHOME/miniconda3/bin/conda init bash