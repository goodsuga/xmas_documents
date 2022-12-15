# Файл для утилит
from pathlib import Path
import tika
tika.initVM()
from tika import parser
import re

import flaml
import pandas as pd
from tqdm import tqdm
import json
from typing import Iterable, List

def get_document_text(path: str) -> str:
    parsed = parser.from_file(path)
    return parsed['content']


def make_train_dataset(data_dir: Path, class_file: Path) -> pd.DataFrame:
    with open(class_file, "r", encoding='utf-8') as read_classes:
        class_data = json.loads(read_classes.read())
    
    rows = [
        {
            "Класс документа": class_data[file.name],
            "Текст документа": get_document_text(str(file))
        }
        for file in tqdm(map(Path, data_dir.glob("*")))
    ]
    return pd.DataFrame(rows)

#CLASS_PATH = r"C:\Users\teberda\Documents\GitHub\xmas_documents\classes.json"
#ds = make_train_dataset(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs"), CLASS_PATH)
#ds.to_parquet("train.pqt")

def clear_texts(texts: Iterable) -> 'List[str]':
    """
    Чистим текст от мусора в виде служебных символов
    """
    allowed_chars = " 0123456789абвгдеёжзийклмонпрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz"
    def preprocess(text):
        clear = text.replace("\n", " ").lower()
        clear = re.sub(" +", " ", clear)
        clear = "".join([c for c in clear if c in allowed_chars])
        return clear
    
    return list(map(preprocess, tqdm(texts)))
