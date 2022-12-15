import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from statistics import mean
from catboost import CatBoostClassifier


def cross_validate_model(model, data, y):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    cv_res = cross_validate(
        model,
        data,
        y,
        cv=cv,
        n_jobs=1,
        scoring={"f1": make_scorer(lambda yt, yp: f1_score(yt, yp, average='macro')),
                    "accuracy": make_scorer(accuracy_score)}
    )
    cv_res = {key: mean(cv_res[key]) for key in cv_res}
    return cv_res


def multinomial_baseline():
    data = pd.read_parquet("train_no_trash.pqt")
    class_map = pd.factorize(data['Класс документа'])
    class_map = {document_class: document_class_index
                for document_class_index, document_class in zip(class_map[0], class_map[1])}

    data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)
    model = make_pipeline(TfidfVectorizer(lowercase=False, analyzer='word', min_df=3), MultinomialNB())
    print(cross_validate_model(model, data['Текст документа'], data['Класс документа (индекс)']))

from sklearn.neighbors import KNeighborsClassifier as KNC
def knc_baseline():
    data = pd.read_parquet("train_no_trash.pqt")
    class_map = pd.factorize(data['Класс документа'])
    class_map = {document_class: document_class_index
                for document_class_index, document_class in zip(class_map[0], class_map[1])}

    data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)
    model = make_pipeline(TfidfVectorizer(lowercase=False, analyzer='word', min_df=3), KNC())
    print(cross_validate_model(model, data['Текст документа'], data['Класс документа (индекс)']))


def catboost_baseline():
    data = pd.read_parquet("train_no_trash.pqt")
    class_map = pd.factorize(data['Класс документа'])
    class_map = {document_class: document_class_index
                for document_class_index, document_class in zip(class_map[0], class_map[1])}

    data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)
    clf = CatBoostClassifier(
        n_estimators=100,
        text_features=["Текст документа"]
    )
    print(cross_validate_model(clf, data[['Текст документа']], data['Класс документа (индекс)']))
    

def make_catboost():
    data = pd.read_parquet("train_no_trash.pqt")
    class_map = pd.factorize(data['Класс документа'])
    class_map = {document_class: document_class_index
                for document_class_index, document_class in zip(class_map[0], class_map[1])}

    data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)
    clf = CatBoostClassifier(
        text_features=["Текст документа"]
    )
    clf.fit(data[['Текст документа']], data['Класс документа (индекс)'])
    clf.save_model("CatboostBaseline.cbm")

import eli5
from sklearn.pipeline import make_pipeline
from IPython import display
def try_explain_catboost():
    data = pd.read_parquet("train_no_trash.pqt")
    class_map = pd.factorize(data['Класс документа'])
    class_map = {document_class: document_class_index
                for document_class_index, document_class in zip(class_map[0], class_map[1])}

    data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)
    clf = make_pipeline(TfidfVectorizer(lowercase=False, analyzer='word', min_df=3), MultinomialNB())
    clf.fit(data['Текст документа'], data['Класс документа (индекс)'])

    display.display(
    a = eli5.show_prediction(
        clf, data['Текст документа'].iloc[0]
    ))
    


if __name__ == "__main__":
    print("MultinomailNB:")
    multinomial_baseline()
    print("KNC:")
    knc_baseline()
    #try_explain_catboost()
    #make_catboost()
    #catboost_baseline()
