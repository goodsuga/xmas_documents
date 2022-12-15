# Файл для утилит
from pathlib import Path
TEST_PATH = r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs\0b4be82b86eff410d69d1d6b5553d220.docx"
import tika
tika.initVM()
from tika import parser

import flaml
import pandas as pd

def get_document_text(path: Path) -> str:
    parsed = parser.from_file(path)
    print(type(parsed['metadata']))
    print(type(parsed['content']))


def make_train_dataset(data_dir: Path) -> pd.DataFrame:
    pass


get_document_text(TEST_PATH)