import os
import pandas as pd
from sqlalchemy import RowMapping
import re
from typing import List
from langchain_core.documents import Document
from konlpy.tag import Okt

okt = Okt()

def create_documents(rows: List[RowMapping]):
    documents = []
    all_texts = get_info_and_texts(rows)
    for (report_info, text) in all_texts:
        text = clean_text(text)
        documents.append(Document(page_content=text, metadata=report_info))
    return documents

def get_info_and_texts(texts: List[RowMapping]):
    texts_total = []
    new_line = "\n"
    for row in texts:
        temp = ""
        paragraph_text = row['paragraph_text']
        tabular_texts = row['tabular_texts']
        image_texts = row['image_texts']
        report_info = {
            'report_id': row['report_id'],
            'company_name': row['company_name'],
            'stockfirm_name': row['stockfirm_name'],
            'report_date': row['report_date']
        }
        
        def merge_texts(arg):
            temp = ""
            new_line = "\n"
            if isinstance(arg, List):
                temp += new_line.join([t for t in arg]) + new_line
            elif isinstance(arg, str) and arg != 'None':
                temp += arg + new_line
            return temp
            
        temp += merge_texts(paragraph_text)
        temp += merge_texts(tabular_texts)
        temp += merge_texts(image_texts)

        texts_total.append((report_info, temp))
    return texts_total


def clean_texts(texts: List[str]):
    import re
    from konlpy.tag import Okt

    okt = Okt()

    # Example stopwords (customize for your domain)
    KOREAN_STOPWORDS = {
        "의", "가", "이", "은", "는", "엔", "에", "들", "와", "과", "도", "를", "으로", "자", "고",
        "수", "다", "로", "가서"  # etc.
    }
    
    cleaned_texts = []

    for text in texts:
        if isinstance(text, Document):
            text = text.page_content
        # Remove special characters: keep only Korean (가-힣), English letters, numbers, and whitespace
        text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

        # Convert to lowercase
        text = text.lower()

        # Morphological analysis with Okt
        tokens = okt.morphs(text, stem=True, norm=True)

        # Remove stopwords
        filtered_tokens = [t for t in tokens if t not in KOREAN_STOPWORDS and t.strip()]

        # Re-join tokens
        cleaned_text = " ".join(filtered_tokens)
        if isinstance(text, Document):
            text.page_content = cleaned_text
        else:
            cleaned_texts.append(cleaned_text)

    return cleaned_texts

def clean_text(text: str):
    # Example stopwords (customize for your domain)
    KOREAN_STOPWORDS = {
        "의", "가", "이", "은", "는", "엔", "에", "들", "와", "과", "도", "를", "으로", "자", "고",
        "수", "다", "로", "가서"  # etc.
    }
    
    # Remove special characters: keep only Korean (가-힣), English letters, numbers, and whitespace
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Morphological analysis with Okt
    tokens = okt.morphs(text, stem=True, norm=True)

    # Remove stopwords
    filtered_tokens = [t for t in tokens if t not in KOREAN_STOPWORDS and t.strip()]

    # Re-join tokens
    cleaned_text = " ".join(filtered_tokens)
    return cleaned_text

def load_xlsx_from(base_dir: str):
    contents = []
    try:
        for i, company_name in enumerate(os.listdir(base_dir)):
            company_path = os.path.join(base_dir, company_name)
            if not os.path.isdir(company_path):
                continue

            for j, excel_file in enumerate(os.listdir(company_path)):
                if not excel_file.endswith(".xlsx"):
                    continue
                report_id = f"{i}_{j}"
                # 증권사명, 작성일 parsing

                # Load Excel file
                excel_path = os.path.join(company_path, excel_file)
                df = pd.read_excel(excel_path)

                # Process each row in the Excel file
                for idx, row in df.iterrows():
                    idx_tabular, idx_image = 0, 0
                    text = row["embedding"]

                    # Generate UUID for paragraph_id
                    paragraph_id = f"{report_id}_{idx}"

                    # TODO: <table> </table>, <img alt= />, <img /> 같은 경우면 <t, <i로 바꾸기
                    tbl_begin_abbr = "<t"
                    tbl_end_abbr = "</t"
                    img_begin_abbr = "<i"
                    img_end_abbr = "/>"
                    
                    tbl_begin = "<table>"
                    tbl_end = "</table>"
                    img_begin = "<img alt="
                    img_end = "/>"

                    # Fix typo
                    text = text.replace("<\\t", tbl_end_abbr)
                    text = text.replace("\\>", img_end_abbr)

                    # unify begin sequence
                    # able> 방지
                    text = text.replace(img_begin, img_begin_abbr)
                    text = text.replace(tbl_begin, tbl_begin_abbr)
                    
                    # Determine content type
                    is_tabular = tbl_begin_abbr in text and tbl_end_abbr in text
                    is_image = img_begin_abbr in text and img_end_abbr in text

                    text = text.replace(img_begin_abbr, img_begin)
                    text = text.replace(img_end_abbr, img_end)
                    text = text.replace(tbl_begin_abbr, tbl_begin)
                    text = text.replace(tbl_end_abbr, tbl_end)

                    # Extract paragraph text
                    if not (is_tabular or is_image):
                        paragraph_text = text
                    elif is_tabular and not is_image:
                        paragraph_text = None
                    elif is_image and not is_tabular:
                        paragraph_text = None
                    else:
                        # TODO: 사실은 이렇지 않을 수 있다
                        # sdlkgnsdlgin <t sdfhsdeg </t <i sdibgvnsdg /> 
                        # sdlkgnsdlgin <i sdibgvnsdg /> <t sdfhsdeg </t
                        # sdlkgnsdlgin <i sdibgvnsdg /> <t sdfhsdeg </t sdbgnsodign
                        # sdlkgnsdlgin <i sdibgvnsdg /> srdibjnslikgvjn <t sdfhsdeg </t sdbgnsodign
                        paragraph_text = text.split(tbl_begin)[0].split(img_begin)[0].strip()

                    # TODO: Multiple tabular/image case
                    if paragraph_text:
                        contents.append(paragraph_text)
                    if is_tabular:
                        # TODO 1: <t dfbdfs </t <t dsfhbdrgh </t 
                        # TODO 2: dfbdfg -> <table>dfbdfg</table>
                        tabular_content = tbl_begin + text.split(tbl_begin)[1].split(tbl_end)[0].strip() + tbl_end
                        contents.append(tabular_content)
                    if is_image:
                        image_content = img_begin + text.split(img_begin)[1].split(img_end)[0].strip() + img_end
                        contents.append(image_content)
        return contents
    except ConnectionError as e:
        print(e)