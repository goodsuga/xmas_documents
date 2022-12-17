# Файл для утилит
from pathlib import Path
import tika

from tika import parser
import re

import pandas as pd
from tqdm import tqdm
import json
from typing import Iterable, List

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import make_pipeline
import optuna
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from functools import partial
from statistics import mean


tika.initVM()

def get_document_text(path: str) -> str:
    parsed = parser.from_file(path)
    return parsed['content']


def make_predict_dataset(data_dir: Path):
    rows = [
        {
            "Класс документа": -1,
            "Текст документа": get_document_text(str(file))
        }
        for file in tqdm(map(Path, data_dir.glob("*")))
    ]
    data = pd.DataFrame(rows)
    data.loc[:, "Текст документа"] = clear_texts(data['Текст документа'])
    return data


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
    data = pd.DataFrame(rows)
    data.loc[:, "Текст документа"] = clear_texts(data['Текст документа'])
    return data


def clear_texts(texts: Iterable) -> 'List[str]':
    """
    Чистим текст от мусора в виде служебных символов
    """
    allowed_chars = " абвгдеёжзийклмонпрстуфхцчшщъыьэюяabcdefghijklmnopqrstuvwxyz"
    def preprocess(text):
        clear = text.replace("\n", " ").lower()
        clear = "".join([c for c in clear if c in allowed_chars])
        clear = re.sub(" +", " ", clear)
        return clear

    return list(map(preprocess, tqdm(texts)))


class OptimizableModelBase:
    def __init__(self, model_class, search_space):
        self.model_class = model_class
        self.search_space = search_space

    def propose_vectorizer(self, trial):
        vec = trial.suggest_categorical("vectorizer", ["Count", "Tfidf"])
        ngram_range=(
                trial.suggest_int("ngram_start", 0, 2),
                max(trial.params['ngram_start'], trial.suggest_int("ngram_end", 1, 3))
        )
        params = {
            "analyzer": trial.suggest_categorical("analyzer", ["word", "char", "char_wb"]),
            "max_df": trial.suggest_float("max_df", 0.5, 1.0),
            "min_df": trial.suggest_float("min_df", 0.01, 0.49),
            "ngram_range": ngram_range
        }
        if vec == "Tfidf":
            params.update(
                {
                    "use_idf": trial.suggest_categorical("use_idf", [True, False]),
                    "norm": trial.suggest_categorical("norm", ["l1", "l2"]),
                    "smooth_idf": trial.suggest_categorical("smooth_idf", [True, False])
                }
            )
        return {
            "Count": CountVectorizer,
            "Tfidf": TfidfVectorizer
        }[vec](**params)

    def rebuild_vectorizer(self, params):
        ngram_range = (params['ngram_start'], params['ngram_end'])
        ngram_range = (params['ngram_start'], max(params['ngram_start'], params["ngram_end"]))
        paramkeys = ["analyzer", "max_df", "min_df"]
        if params['vectorizer'] == 'Tfidf':
            paramkeys.extend(["use_idf", "norm", "smooth_idf"])
        passed_params = {paramkey: params[paramkey] for paramkey in paramkeys}
        passed_params['ngram_range'] = ngram_range
        return {
            "Count": CountVectorizer,
            "Tfidf": TfidfVectorizer
        }[params['vectorizer']](**passed_params)

    def propose_model(self, trial):
        params = {}
        for param in self.search_space:
            params[param] = self.search_space[param](trial)
        model = self.model_class()
        model.set_params(**params)
        return model

    def rebuild_model(self, params):
        passed_params = {param: params[param] for param in self.search_space}
        model = self.model_class()
        model.set_params(**passed_params)
        return model

    def get_optimization_objective(self, X, y, cv):
        def objective(trial):
            model = self.propose_model(trial)
            vec = self.propose_vectorizer(trial)
            clf = make_pipeline(vec, model)
            return mean(
                cross_val_score(
                    clf, X, y, cv=cv,
                    scoring=make_scorer(lambda yt, yp: f1_score(yt, yp, average='macro')),
                    n_jobs=-1
                )
            )
        return objective
