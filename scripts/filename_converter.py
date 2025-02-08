import json
import os
import pathlib
from typing import List


def convert_names(filename: str) -> str:
    converter_path = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "data/filename_converter.json"
    )
    with open(converter_path, "r") as f:
        converter = json.load(f)
        original: List[str] = converter["original"]
        new: List[str] = converter["new"]

        try:
            print(new[original.index(filename)])
            return new[original.index(filename)]
        except ValueError:
            print(original[new.index(filename)])
            return original[new.index(filename)]
        except Exception as e:
            print(f"Unexpected {e=}, {type(e)=}")
            raise


def convert_to_new(filename: str) -> str:
    converter_path = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "data/filename_converter.json"
    )
    with open(converter_path, "r") as f:
        converter = json.load(f)
        original: List[str] = converter["original"]
        new: List[str] = converter["new"]

        try:
            print(new[original.index(filename)])
            return new[original.index(filename)]
        except ValueError:
            return filename
        except Exception as e:
            print(f"Unexpected {e=}, {type(e)=}")
            raise


def convert_datasets():
    converter_path = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "data/filename_converter.json"
    )
    datasets_path = os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(), "data/datasets.json"
    )
    with open(converter_path, "r") as f:
        converter = json.load(f)
        with open(datasets_path, "r") as f:
            datasets = json.load(f)
            for row in datasets["qa_sets"]:
                row["source"] = convert_to_new(row["source"])
            with open("new_datasets.json", "w", encoding="utf-8") as f:
                json.dump(datasets, f, ensure_ascii=False, default=str, indent=4)


if __name__ == "__main__":
    # convert_datasets()
    print("종료하시려면 Ctrl + C를 입력하세요.")
    while True:
        query = input("변환할 구버전/신버전 파일명을 입력하세요: ")
        convert_names(query)
