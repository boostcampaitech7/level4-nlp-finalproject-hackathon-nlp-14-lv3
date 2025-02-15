# 증권사 자료 기반 주식 LLM 서비스 개발

<img src="https://img.shields.io/badge/python-%233776AB.svg?&style=flat-square&logo=python&logoColor=white" /> <img src="https://img.shields.io/badge/postgresql-%23336791.svg?&style=flat-square&logo=postgresql&logoColor=white" /> <img src = "https://img.shields.io/badge/langchain-1C3C3C?style=flat-square&logo=html5&logoColor=white" />
수정예정

#### 다양한 산업에서 LLM 도입을 시도하고 있습니다. 할루시네이션 문제로 어려움을 겪고 있으나 RAG와 같은 기술을 도입하여 할루시네이션 제거하는데 연구 중에 있습니다. 본 프로젝트는 PDF 파싱을 통한 데이터 구축하고, RAG 기술을 활용하여 할루시네이션 현상 최소화하려 합니다. 


## 팀원 소개
- 표로 작성


## 서비스 아키텍쳐
- 이미지 삽입


## Conda 설치 및설정
- 아래 명령어 실행시 dev 브랜치 및 데이터 레포 복제까지 실행
```zsh
apt-get update &&
apt-get upgrade &&
apt-get install git &&

conda init &&
mkdir -p /data/ephemeral/conda/pkgs && mkdir -p /data/ephemeral/conda/envs && mkdir -p /data/ephemeral/tmp && mkdir -p /data/ephemeral/pip/cache && echo -e "pkgs_dirs:\n - /data/ephemeral/conda/pkgs\nenvs_dirs:\n - /data/ephemeral/conda/envs" >> ~/.condarc && echo -e "\n# Custom environment variables\nexport TMPDIR=/data/ephemeral/tmp\nexport CONDA_PKGS_DIRS=/data/ephemeral/conda/pkgs\nexport CONDA_ENVS_DIRS=/data/ephemeral/conda/envs\nexport PIP_CACHE_DIR=/data/ephemeral/pip/cache" >> ~/.bashrc &&
source ~/.bashrc &&
conda create -n langchain python=3.10 --yes &&
conda activate langchain &&

# GIT SSH 키 발급 후에 진행
ssh-keygen &&
cat /root/.ssh/id_rsa.pub &&
git clone git@github.com:boostcampaitech7/level4-nlp-finalproject-hackathon-nlp-14-lv3.git &&
cd level4-nlp-finalproject-hackathon-nlp-14-lv3 &&
git submodule update --init &&
git checkout dev &&
pip install -r ./src/requirements.txt
```


## GCP CLI 설정

### GCP 관련 필요 패키지 설치
```bash
pip install --upgrade google-api-python-client
```
```bash
pip install "cloud-sql-python-connector[pg8000]"\
```
Google Cloud Platform에 들어가서 프로렉트와 SQL 인스턴스를 생성합니다.

### 로컬 개발 환경에 ADC 설정
```bash
./google-cloud-sdk/bin/gcloud init
```
```bash
./google-cloud-sdk/bin/gcloud auth application-default login
```

### 선택사항
.env 파일을 생성하여 [PYTHON으로 GCP 연결](https://cloud.google.com/sql/docs/postgres/samples/cloud-sql-postgres-sqlalchemy-connect-connector?hl=ko)에서 사용되는 민감한 정보를 관리합니다.


## DATA

### PDF 파싱
한줄 설명
```bash
# BASH에서 어떻게 작동하면 되는지
```

### 데이터 처리
한줄 설명
```bash
# BASH에서 어떻게 작동하면 되는지
```


## RAG

### rag 실행
한줄 설
```bash
# BASH에서 어떻게 작동하면 되는지
```
