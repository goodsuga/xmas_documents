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
from sklearn.model_selection import StratifiedKFold
import plotly.graph_objects as go
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
        clear = "".join([c for c in clear if c in allowed_chars])
        clear = re.sub(" +", " ", clear)
        return clear
    
    return list(map(preprocess, tqdm(texts)))


def explain_instance(model, class_map, instance_str, n_features=5, document_name=None):
    explainer = LimeTextExplainer(class_names=list(class_map.keys()))
    exp = explainer.explain_instance(instance_str, 
                                    model.predict_proba,
                                    labels=list(class_map.values()),
                                    num_features=200,
                                    num_samples=200)
    
    labels = []
    values = model.predict_proba([instance_str])[0]
    for key in class_map:
        imps = pd.DataFrame(exp.as_list(label=class_map[key]), columns=['Слово', f'Вклад'])
        imps = imps.sort_values(by="Вклад", ascending=False).iloc[:n_features]
        joined_words = ";".join(imps['Слово'])
        labels.append(
            f"{key.replace('Договоры для акселератора/', '')}<br>[{joined_words}]"
        )

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
    return fig


from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.pipeline import make_pipeline
import optuna
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from functools import partial


def _rebuild_vec(params):
    vec = params["vectorizer"]
    analyzer = params["analyzer"]
    max_df = params["max_df"]
    min_df = params["min_df"]
    ngram_range=(
                params["ngram_start"],
                max(params['ngram_start'], params["ngram_end"])
            )

    if vec == "Count":
        vec = CountVectorizer(
            lowercase=False,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df
        )
    elif vec == "Tfidf":
        vec = TfidfVectorizer(
            lowercase=False,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            use_idf=params["use_idf"],
            norm=params["norm"],
            smooth_idf=params["smooth_idf"]
        )
    return vec


def _rebuild_logistic(params):
    print(params)
    model = LogisticRegression(C=params['C'],
                               max_iter=100,
                                class_weight='balanced',
                                solver=params["solver"])
    return model

def _rebuild_sgd(params):
    print(params)
    model = SGDClassifier(
        loss=params["loss"],
        penalty=params["penalty"],
        class_weight="balanced"
    )
    return model

from statistics import mean
def _propose_vec(trial: optuna.Trial):
    vec = trial.suggest_categorical("vectorizer", ["Count", "Tfidf"])
    analyzer = trial.suggest_categorical("analyzer", ["word", "char", "char_wb"])
    max_df=trial.suggest_float("max_df", 0.5, 1.0)
    min_df=trial.suggest_float("min_df", 0.01, 0.49)
    ngram_range=(
                trial.suggest_int("ngram_start", 0, 2),
                max(trial.params['ngram_start'], trial.suggest_int("ngram_end", 1, 3))
            )

    if vec == "Count":
        vec = CountVectorizer(
            lowercase=False,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df
        )
    elif vec == "Tfidf":
        vec = TfidfVectorizer(
            lowercase=False,
            analyzer=analyzer,
            ngram_range=ngram_range,
            max_df=max_df,
            min_df=min_df,
            use_idf=trial.suggest_categorical("use_idf", [True, False]),
            norm=trial.suggest_categorical("norm", ["l1", "l2"]),
            smooth_idf=trial.suggest_categorical("smooth_idf", [True, False])
        )
    return vec


def _logistic_reg_objective(cv: RepeatedStratifiedKFold, X, y, trial: optuna.Trial) -> float:
    vec = _propose_vec(trial)
    solver = trial.suggest_categorical("solver", ["newton-cg", "sag", "saga", "lbfgs"])
    
    model = LogisticRegression(C=trial.suggest_float("C", 0.001, 10.0),
                                max_iter=500,
                                class_weight='balanced',
                                solver=solver)
    
    pipe = make_pipeline(vec, model)
    print(trial.params)
    return mean(cross_val_score(
        pipe, X, y, cv=cv,
        scoring=make_scorer(lambda yt, yp: f1_score(yt, yp, average='macro')),
        n_jobs=-1
    ))


def _sgd_objective(cv: RepeatedStratifiedKFold, X, y, trial: optuna.Trial) -> float:
    vec = _propose_vec(trial)
    losses = ['log_loss',
              'modified_huber',
              'huber'
    ]
    model = SGDClassifier(
        loss=trial.suggest_categorical("loss", losses),
        penalty=trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"]),
        class_weight="balanced"
    )
    
    pipe = make_pipeline(vec, model)
    return mean(cross_val_score(
        pipe, X, y, cv=cv,
        scoring=make_scorer(lambda yt, yp: f1_score(yt, yp, average='macro')),
        n_jobs=-1
    ))


def configure_models(train_df,
                     y,
                     cv,
                     allowed_models="all",
                     max_iters=150):
    
    supported_names = ["linear", "sgd", "knc", "bayes"]
    if isinstance(allowed_models, list):
        assert all([name in supported_names for name in allowed_models])
        assert isinstance(max_iters, dict)
        models = allowed_models
    else:
        models = supported_names
        max_iters = {name: max_iters for name in models}
    
    studies = {}
    for model in models:
        study = optuna.create_study(direction='maximize')
        if model == "linear":
            obj = partial(_logistic_reg_objective, cv, train_df, y)
        elif model == "sgd":
            obj = partial(_sgd_objective, cv, train_df, y)

        study.optimize(obj, n_trials=max_iters[model], show_progress_bar=True)
        studies[model] = study
        print(f"BEST PARAMS FOR {model} ==> {study.best_params}")
    
    return studies
    


# data = pd.read_parquet("train_no_trash.pqt")
# class_map = pd.factorize(data['Класс документа'])
# class_map = {document_class: document_class_index
#             for document_class_index, document_class in zip(class_map[0], class_map[1])}

# data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)

# import pymorphy2
# analyzer = pymorphy2.MorphAnalyzer()

# normform = lambda word: analyzer.parse(word)[0].normal_form

# data.loc[:, 'Текст документа'] = data['Текст документа'].apply(
#     lambda text: " ".join(map(normform, re.sub(" +", " ", text).split()))
# )

# from sklearn.linear_model import LogisticRegressionCV
# from sklearn.neighbors import KNeighborsClassifier as KNC
# from sklearn.calibration import CalibratedClassifierCV
# from sklearn.model_selection import StratifiedKFold

# model = make_pipeline(TfidfVectorizer(lowercase=False, analyzer='word', min_df=3),
#     LogisticRegressionCV())
# model.fit(data['Текст документа'], data['Класс документа (индекс)'])
# class_names = list(class_map.keys())
# for i in range(0, 5):
#     instance_str = data['Текст документа'].iloc[i]
#     explain_instance(model, class_names, instance_str, document_name="Имя документа")