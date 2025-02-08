import json
import os

import pandas as pd

# 외국인 지분율 추가해서 보고서 첫 페이지의 요약자료까지 추가? 아니면 요약자료만?
# "영업실적"
keywords = [
    "재무상태표",
    "대차대조표",
    "손익계산서",
    "포괄손익계산서",
    "현금흐름표",
    "주요투자지표",
    "주요 투자지표",
    "주요지표",
    "투자지표",
    "재무비율",
    "주당지표",
    "Statement of comprehensive income",
    "Valuations/profitability/stability",
    "Statement of financial position",
    "Cash flow statement",
]
prefixes = ["#", "# ", "**", "** "]
middles = ["", "예상"]
anti_prefixes = [".", ". "]
keywords_sharp = [
    prefix + middle + keyword
    for prefix in prefixes
    for middle in middles
    for keyword in keywords
]
anti_keywords = [
    anti_prefix + keyword for anti_prefix in anti_prefixes for keyword in keywords_sharp
]

# 대차대조표, 재무상태표, Balance Sheet,
bs_keywords = {"keywords": ["자산총계", "부채총계"], "title": "대차대조표"}

# 손익계산서, 포괄손익계산서, Statement of comprehensive income
ci_keywords = {
    "keywords": ["영업이익", "순이익", "EBITDA"],
    "excepts": ["PER", "ROA", "실적추정", "트렌드"],
    "title": "손익계산서",
}

# 현금흐름표, cash flow statement
cf_keywords = {
    "keywords": ["FCF", "Free Cash Flow", "영업활동현금흐름", "기초현금", "기말현금"],
    "title": "현금흐름표",
}

# 주요투자지표, 주요지표, 투자지표
kf_keywords = {
    "keywords": ["EPS", "Valuation", "BPS", "ROA", "EBITDA"],
    "excepts": ["적정가치", "Stock Data", "Investment Fundamental"],
    "title": "주요투자지표",
}

new_keywords = [bs_keywords, ci_keywords, cf_keywords, kf_keywords]

results = dict()


def process_excel_and_insert_data(base_dir):
    """
    Processes the Excel files and inserts data into the database.
    """
    # Iterate through each company directory
    for company_name in os.listdir(base_dir):
        company_path = os.path.join(base_dir, company_name)
        if not os.path.isdir(company_path):
            continue
        # Process each Excel file in the company directory
        for excel_file in os.listdir(company_path):
            if excel_file.endswith(".xlsx"):
                # Extract stockfirm_name and date from filename
                _, stockfirm_name, date_part = os.path.splitext(excel_file)[0].split(
                    "_"
                )
                report_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:]}"
                report_date = str(pd.to_datetime(report_date).date())

                # Insert into report table
                company_dict = results.setdefault(company_name, {})
                stockfirm_dict = company_dict.setdefault(stockfirm_name, {})
                report_dict = stockfirm_dict.setdefault(
                    report_date,
                    {
                        "대차대조표": [],
                        "손익계산서": [],
                        "현금흐름표": [],
                        "주요투자지표": [],
                        "duplicated": [],
                    },
                )

                # Load Excel file
                excel_path = os.path.join(company_path, excel_file)
                df = pd.read_excel(excel_path)

                # Process each row in the Excel file
                cnt = 0
                found = []
                found_sharp = []
                for idx, row in df.iterrows():
                    text = row["text"]
                    if any([text.find(keyword) != -1 for keyword in keywords]):
                        found.append(text)
                    if any(
                        [text.find(keyword) != -1 for keyword in keywords_sharp]
                    ) and all(
                        [text.find(akeyword) == -1 for akeyword in anti_keywords]
                    ):
                        found_sharp.append(text)
                    matched_titles = []
                    # Check each keyword group
                    for item in new_keywords:
                        group_keywords = item["keywords"]
                        title = item["title"]

                        excepts = item.get("excepts", [])
                        if any(ex in text for ex in excepts):
                            continue

                        # Count how many distinct keywords from this group appear in the text
                        count = 0
                        for kw in group_keywords:
                            if kw in text:
                                count += 1

                        # If 2 or more keywords of this group appear in the text, it's a match
                        if count >= 2:
                            matched_titles.append(title)
                            cnt += 1

                    # Based on the matched titles, insert the text into the proper list.
                    if len(matched_titles) == 1:
                        report_dict[matched_titles[0]].append(text)
                    elif len(matched_titles) > 1:
                        print(f"[ALERT] Text matches multiple groups: {matched_titles}")
                        report_dict["duplicated"].append(text)

    with open("results.json", "w", encoding="utf-8") as f:
        # Using default=str to handle any non-serializable objects (like dates)
        json.dump(results, f, ensure_ascii=False, default=str, indent=4)


def is_financial_position(text: str):
    if text.find("유동자산") != -1 and text.find("비유동자산") != -1:
        return True
    return False


if __name__ == "__main__":
    BASE_DIR = "data/랩큐"
    process_excel_and_insert_data(BASE_DIR)


# def get_texts_with_keywords(texts: List[str]) -> List[str]:
#     # TODO
#     # for each text in texts, check if two or more keywords. keywords appears for keywords in new_keywords.
#     # if text have two or more keywords, then append it to dict[keyword.title]
#     # If there are more than one keywords that matches, make alert with print and save it as dict["duplicated"].


# zzzzzzzzzzzzzzzzzzzzzzz
# Statement of comprehensive income
# Valuations/profitability/stability
# Statement of financial position
# Cash flow statement

# # n. 재무상태표
# 재무비율
# Income Statement


# Balance Sheet
# Key Financial Data

# 예상 포괄손익
# 예상 재무상태표
# DuPont analysis
# Key Ratios
