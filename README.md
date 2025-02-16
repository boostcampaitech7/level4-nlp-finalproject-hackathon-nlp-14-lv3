# 증권사 자료 기반 주식 LLM 서비스 개발

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi) ![Postgres](https://img.shields.io/badge/postgres-%23316192.svg?style=for-the-badge&logo=postgresql&logoColor=white) ![ChatGPT](https://img.shields.io/badge/chatGPT-74aa9c?style=for-the-badge&logo=openai&logoColor=white) ![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white) ![Langchain](https://img.shields.io/badge/langchain-%1c3c35.svg?style=for-the-badge&logo=langchain&logoColor=white)

#### 다양한 산업에서 LLM 도입을 시도하고 있습니다. 할루시네이션 문제로 어려움을 겪고 있으나 RAG와 같은 기술을 도입하여 할루시네이션 제거하는데 연구 중에 있습니다. 본 프로젝트는 PDF 파싱을 통한 데이터 구축하고, RAG 기술을 활용하여 할루시네이션 현상 최소화하는 것을 목표로 합니다. 

## 팀원 소개
- 수정 시 **'정사각형'** 이미지 넣어주시고, 해당 문구는 삭제해주세요!
- 양식 : ![이미지](링크)
  
| ![KakaoTalk_20241224_115814106](https://github.com/user-attachments/assets/338cb43f-8d34-4d4b-bc8c-fb66f48d8a5c) | ![KakaoTalk_20241224_115814106](https://github.com/user-attachments/assets/338cb43f-8d34-4d4b-bc8c-fb66f48d8a5c) | ![KakaoTalk_20241224_115814106](https://github.com/user-attachments/assets/7c0a0b02-808a-4d41-a17d-e66d3f0728bd) | ![KakaoTalk_20241224_115814106](https://github.com/user-attachments/assets/a13a1c7c-7346-43f6-b949-cfb1204fc695) | ![KakaoTalk_20241224_115814106](https://github.com/user-attachments/assets/933002f0-c5fe-44aa-ba9d-de7acc07ecd4) |
| :---: | :---: | :---: | :---: | :---: |  
| 김경인 | 김준섭 | 김채연 | 오승범 | 이시온 |
| DATA & DB | Backend | DATA & DB | DATA & RAG | RAG |


## 서비스 아키텍쳐
- 이미지를 삽입하거나 색상을 구분하여 수정했을 경우 수정해서 넣어주세요! 수정 완료한 뒤에는 해당 문구를 삭제해주세요.

![image](https://github.com/user-attachments/assets/cb7e039d-d1a0-4320-a855-4b36c4a8ed0d)



## Conda 설치 및설정
- 해당 Repo를 clone하고, 아래 명령어 실행시 dev 브랜치 및 데이터 레포 복제까지 실행됩니다.
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
pip install "cloud-sql-python-connector[pg8000]"
```
#### Google Cloud Platform에 들어가서 프로렉트와 SQL 인스턴스를 생성합니다.

### 로컬 개발 환경에 ADC 설정
```bash
./google-cloud-sdk/bin/gcloud init
```
```bash
./google-cloud-sdk/bin/gcloud auth application-default login
```

### 선택사항
.env 파일을 생성하여 [PYTHON으로 GCP 연결 코드](https://cloud.google.com/sql/docs/postgres/samples/cloud-sql-postgres-sqlalchemy-connect-connector?hl=ko)에서 사용되는 민감한 정보를 관리합니다.


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

### RAG 실행
한줄 설명
```bash
# BASH에서 어떻게 작동하면 되는지
```

##### 최종 수정 날짜 : 250216
