import os

from dotenv import load_dotenv

from google.cloud import storage
from google.oauth2 import service_account

import pymupdf4llm
import zipfile

load_dotenv()

# Read all of folders
def company_path_dictionary(base_dir):
    company_paths = {}
    
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
    # 폴더인지 확인 (파일 제외)
        if os.path.isdir(folder_path):
            company_paths[folder_name] = folder_path
            
    return company_paths

# Change file name
def change_file_name(company_paths):
    llama_reader = pymupdf4llm.LlamaMarkdownReader()
    file_name_map = {}
    
    for company_name, company_path in company_paths.items():
        for file_name in os.listdir(company_path):
            file_path = os.path.join(company_path, file_name)

            # PDF 파일인지 확인
            if file_name.endswith('.pdf'):
                try:
                    # PDF 파일의 메타데이터 추출
                    llama_docs = llama_reader.load_data(file_path)
                    metadata = llama_docs[0].to_dict()
                    
                    securities_firms_dict = {
                                            "교보증권": ["kyobo", "교보증권"],
                                            "SK증권": ["sk", "SK증권"],
                                            "미래에셋증권": ["miraeasset", "미래에셋증권"],
                                            "IBK투자증권": ["ibk", "IBK투자증권"],
                                            "한화투자증권": ["hanwha", "한화투자증권"],
                                            "신한투자증권": ["shinhan", "신한투자증권"],
                                            "하나증권": ["hana", "하나증권"],
                                            "유안타증권": ["yuanta", "유안타증권"],
                                            "유진투자증권": ["eugene", "유진투자증권"],
                                            "키움증권": ["kiwoom", "키움증권"],
                                            "DS투자증권": ["ds", "DS투자증권"],
                                            "iM증권": ["im", "iM증권"],
                                            "메리츠증권": ["meritz", "메리츠증권"],
                                            "삼성증권": ["samsung", "삼성증권"],
                                            "대신증권": ["daishin", "대신증권"],
                                            "DB금융투자": ["dbfinancial", "DB금융투자"],
                                            "NH투자증권": ["nh", "NH투자증권"],
                                            "현대차증권": ["hyundai", "현대차증권"],
                                            "LS증권": ["ls", "LS증권"],
                                            "한국신용평가": ["koreacreditrating", "한국신용평가"]
                                        }

                    
                    # 파일명에서 증권사 이름 먼저 찾기
                    securities_name = None
                    file_name_lower = file_name.lower()
                    for firm in securities_firms_dict:
                        if firm.lower() in file_name_lower:
                            securities_name = firm
                            break

                    # 파일명에서 찾지 못하면 PDF 본문에서 검색
                    if not securities_name:
                        pdf_text = "\n".join([str(page) for page in llama_docs]).lower()
                    
                        for firm in securities_firms_dict:
                            if firm.lower() in pdf_text:
                                securities_name = firm
                                break
                    
                    # 여전히 없으면 기본값 사용
                    if not securities_name:
                        securities_name = "UnknownSecurities"

                    # 생성 날짜 추출 및 포맷팅 (YYYYMMDD 형식)
                    creation_date = metadata.get('metadata', {}).get('creationDate', '00000000')
                    formatted_date = creation_date[2:10] if len(creation_date) >= 8 else '00000000'

                    # 새 파일명 생성
                    new_file_name = f"{company_name}_{securities_name}_{formatted_date}.pdf"
                    new_file_path = os.path.join(company_path, new_file_name)

                    file_name_map[file_name.replace('.pdf', '')] = new_file_name.replace('.pdf', '')

                    # 파일명 변경
                    os.rename(file_path, new_file_path)
                    print(f"Renamed: {file_name} -> {new_file_name}")

                    # `_images` 폴더도 동일하게 이름 변경
                    old_images_folder = os.path.join(company_path, file_name.replace('.pdf', '_images'))
                    new_images_folder = os.path.join(company_path, new_file_name.replace('.pdf', '_images'))
                    
                    if os.path.exists(old_images_folder):
                        os.rename(old_images_folder, new_images_folder)
                        print(f"Renamed Images Folder: {old_images_folder} -> {new_images_folder}")

                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    for company_name, company_path in company_paths.items():
        for file_name in os.listdir(company_path):
            if file_name.endswith(('.xls', '.xlsx', '.csv')):
                base_name, ext = os.path.splitext(file_name)

                if base_name in file_name_map:
                    new_file_name = file_name_map[base_name] + ext
                    new_file_path = os.path.join(company_path, new_file_name)
                    file_path = os.path.join(company_path, file_name)

                    os.rename(file_path, new_file_path)
                    print(f"Renamed Excel: {file_name} -> {new_file_name}")
   

# GCP Storage에 단일 파일 업로드       
def upload_file(bucket_name, local_file_path, destination_blob_name):
    credentials = service_account.Credentials.from_service_account_file(
        os.environ['JSON_FILE']
    )
    storage_client = storage.Client(credentials=credentials)
    
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        print(f"Uploaded: {local_file_path} -> {destination_blob_name}")
    except Exception as e:
        print(f"Error uploading {local_file_path}: {e}")

# 파일 수정 후 Cloud Storage에 업로드

def upload_files_cloud_storage():
    """
    회사 폴더 내 파일을 타입별로 분류하여 GCP Storage에 업로드하는 함수.
    """
    for company_name, company_path in company_paths.items():
        print(f"Processing files for company: {company_name}")

        pdf_files = {}  # PDF 파일명 저장 (이미지 압축 시 사용)

        # 폴더 내 모든 파일 탐색
        for root, _, files in os.walk(company_path):
            for file in files:
                local_file_path = os.path.join(root, file)

                # 파일 확장자에 따라 GCP 경로 결정
                if file.endswith('.pdf'):
                    destination_blob_name = f"data/{company_name}/reports/{file}"
                    upload_file(bucket_name, local_file_path, destination_blob_name)
                    pdf_files[file.replace('.pdf', '')] = file
                elif file.endswith(('.xls', '.xlsx', '.csv')):
                    destination_blob_name = f"data/{company_name}/excel/{file}"
                    upload_file(bucket_name, local_file_path, destination_blob_name)
                    
                else:
                    # 지원하지 않는 파일 형식은 건너뜀
                    print(f"Skipping unsupported file: {local_file_path}")
                    continue

        # _images 폴더를 찾아 압축 후 업로드
        for pdf_name, pdf_file in pdf_files.items():
            images_folder = os.path.join(company_path, f"{pdf_name}_images")
            
            if os.path.exists(images_folder):  # _images 폴더 존재 여부 확인
                image_files = [os.path.join(images_folder, img) for img in os.listdir(images_folder)
                               if img.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
                
                if image_files:  # 이미지 파일이 있는 경우에만 ZIP 생성
                    zip_file_name = os.path.join(company_path, f"{pdf_name}_images.zip")
                    
                    with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        for img_file in image_files:
                            zipf.write(img_file, os.path.basename(img_file))  # ZIP 내부에서는 파일명만 유지

                    # 압축된 ZIP 파일 업로드 (GCP 경로: data/{company_name}/images/{pdf_name}_images.zip)
                    destination_blob_name = f"data/{company_name}/images/{pdf_name}_images.zip"
                    upload_file(bucket_name, zip_file_name, destination_blob_name)

                    print(f"Uploaded {pdf_name}_images.zip to {destination_blob_name}")
        
        print(f"All files processed for company: {company_name}")

if __name__ == "__main__":
    BASE_DIR = "data/랩큐"
    BUCKET_DIR = "data/"
    bucket_name = "nlp14" 
    
    company_paths = company_path_dictionary(BASE_DIR)
    
    change_file_name(company_paths)
    
    upload_files_cloud_storage(bucket_name, company_paths)