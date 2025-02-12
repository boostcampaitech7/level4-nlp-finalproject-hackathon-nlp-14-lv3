import json
import os


class RuleRetriever:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, "rule"):
            self.rule = None
            self.keywords = [
                {"keywords": ["자산총계", "부채총계"], "title": "대차대조표"},
                {
                    "keywords": ["영업이익", "순이익", "EBITDA"],
                    "excepts": ["PER", "ROA", "실적추정", "트렌드"],
                    "title": "손익계산서",
                },
                {
                    "keywords": [
                        "FCF",
                        "Free Cash Flow",
                        "영업활동현금흐름",
                        "기초현금",
                        "기말현금",
                    ],
                    "title": "현금흐름표",
                },
                {
                    "keywords": ["EPS", "Valuation", "BPS", "ROA", "EBITDA"],
                    "excepts": ["적정가치", "Stock Data", "Investment Fundamental"],
                    "title": "주요투자지표",
                },
            ]

    def load_rule(self, rule_path: str = "data/results.json"):
        rule_path = os.path.join(os.getcwd(), rule_path)

        if self.rule is None:
            with open(rule_path, "r") as f:
                self.rule = json.load(f)
                self.companies = self.rule.keys()

    def run_retrieval(self, query: str):
        retrieved = []
        candidates = []
        companies = []
        for company in self.companies:
            if query.find(company) == -1:
                continue
            companies.append(company)
            temp = self.rule[company]
            firms_found = []
            for firm in temp.keys():
                if query.find(firm) != -1:
                    firms_found.append(firm)
            if len(firms_found) > 0:
                dates_found = []
                for date in temp[firm].keys():
                    if query.find(date) != -1:
                        dates_found.append(date)
                if len(dates_found) > 0:
                    [candidates.append(temp[firm][date]) for date in dates_found]
                else:
                    candidates.append(temp[firm][date])
            else:
                firm, date = find_company_date(temp)
                candidates.append(temp[firm][date])

        for keywords in self.keywords:
            for keyword in keywords["keywords"]:
                if query.find(keyword) == -1:
                    continue
                for candidate in candidates:
                    retrieved.append(candidate[keywords["title"]])

        return companies, retrieved


def find_company_date(data):
    """
    Given a nested dict with the following structure:
      {
         company_name: {
             date_string: {
                 key1: list,
                 key2: list,
                 key3: list,
                 key4: list
             },
             ...
         },
         ...
      }

    This function returns a tuple (company_name, date_string) for the first inner dict
    where all 4 lists are non-empty. If none exist, it returns the (company_name, date_string)
    for the inner dict that has the highest total number of elements across the 4 lists.
    """
    best_company = None
    best_date = None
    best_total = -1

    # Iterate over companies and their dates (order is the dict order)
    for company, dates in data.items():
        for date, inner in dates.items():
            # Check if all four keys have a non-empty list.
            # (We assume here that inner always has exactly 4 keys.)
            if all(inner[key] for key in inner):
                # Found one that meets the condition: return immediately.
                return company, date

            # Otherwise, count the total number of elements in all lists.
            total = sum(len(inner[key]) for key in inner)
            if total > best_total:
                best_total = total
                best_company = company
                best_date = date

    return best_company, best_date
