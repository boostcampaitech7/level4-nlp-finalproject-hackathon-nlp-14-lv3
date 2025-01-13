# level4-nlp-finalproject-hackathon-nlp-14-lv3


## 설치


- 아래 명령어 실행시 dev 브랜치 및 데이터 레포 복제까지 실행
```zsh
git clone --branch dev --recurse-submodules https://github.com/boostcampaitech7/level4-nlp-finalproject-hackathon-nlp-14-lv3.git final &&
conda init &&
mkdir -p /data/ephemeral/conda/pkgs && mkdir -p /data/ephemeral/conda/envs && mkdir -p /data/ephemeral/tmp && mkdir -p /data/ephemeral/pip/cache && echo -e "pkgs_dirs:\n - /data/ephemeral/conda/pkgs\nenvs_dirs:\n - /data/ephemeral/conda/envs" >> ~/.condarc && echo -e "\n# Custom environment variables\nexport TMPDIR=/data/ephemeral/tmp\nexport CONDA_PKGS_DIRS=/data/ephemeral/conda/pkgs\nexport CONDA_ENVS_DIRS=/data/ephemeral/conda/envs\nexport PIP_CACHE_DIR=/data/ephemeral/pip/cache" >> ~/.bashrc &&
source ~/.bashrc &&
conda create -n langchain python=3.10 --yes &&
conda activate langchain &&
pip install -r /data/ephemeral/home/final/src/requirements.txt
```