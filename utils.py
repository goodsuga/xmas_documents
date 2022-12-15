# Файл для утилит
from pathlib import Path
import tika
tika.initVM()
from tika import parser

import flaml
import pandas as pd
from tqdm import tqdm
import json

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

CLASS_PATH = r"C:\Users\teberda\Documents\GitHub\xmas_documents\classes.json"
print(make_train_dataset(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs"), CLASS_PATH))
