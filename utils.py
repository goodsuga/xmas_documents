# Файл для утилит
from pathlib import Path
import tika

from tika import parser
import re

import pandas as pd
from tqdm import tqdm
import json
from typing import Iterable, List

from lime.lime_text import LimeTextExplainer
from lime import lime_text
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

tika.initVM()

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

import plotly.graph_objects as go
def explain_instance(model, class_names, instance_str, n_features=5, document_name=None):
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(instance_str, 
                                    model.predict_proba,
                                    labels=list(class_map.values()),
                                    num_features=200)
    
    labels = []
    values = []
    for key in class_map:
        imps = pd.DataFrame(exp.as_list(label=class_map[key]), columns=['Слово', f'Вклад'])
        imps = imps.sort_values(by="Вклад", ascending=False).iloc[:n_features]
        joined_words = ";".join(imps['Слово'])
        labels.append(
            f"{key.replace('Договоры для акселератора/', '')}<br>[{joined_words}]"
        )
        values.append(imps['Вклад'].sum())

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=values,
            theta=labels,
            fill='toself'
        )
    )
    if document_name is not None:
        fig.update_layout(title=document_name, title_x=0.5, font=dict(size=12))
    fig.show()


data = pd.read_parquet("train_no_trash.pqt")
class_map = pd.factorize(data['Класс документа'])
class_map = {document_class: document_class_index
            for document_class_index, document_class in zip(class_map[0], class_map[1])}

data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)

import pymorphy2
analyzer = pymorphy2.MorphAnalyzer()

normform = lambda word: analyzer.parse(word)[0].normal_form

data.loc[:, 'Текст документа'] = data['Текст документа'].apply(
    lambda text: " ".join(map(normform, re.sub(" +", " ", text).split()))
)

from sklearn.linear_model import LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold

model = make_pipeline(TfidfVectorizer(lowercase=False, analyzer='word', min_df=3),
    LogisticRegressionCV())
model.fit(data['Текст документа'], data['Класс документа (индекс)'])
class_names = list(class_map.keys())
for i in range(0, 5):
    instance_str = data['Текст документа'].iloc[i]
    explain_instance(model, class_names, instance_str, document_name="Имя документа")