import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.naive_bayes import MultinomialNB, CategoricalNB, GaussianNB, BernoulliNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold
from statistics import mean

def multinomial_baseline():
    data = pd.read_parquet("train_no_trash.pqt")
    class_map = pd.factorize(data['Класс документа'])
    class_map = {document_class: document_class_index
                for document_class_index, document_class in zip(class_map[0], class_map[1])}

    data['Класс документа (индекс)'] = data['Класс документа'].apply(class_map.get)
    print(data)

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)
    model = make_pipeline(TfidfVectorizer(lowercase=False, analyzer='word', min_df=3), MultinomialNB())
    cv_res = cross_validate(
        model,
        data['Текст документа'],
        data['Класс документа (индекс)'],
        cv=cv,
        n_jobs=-1,
        scoring={"f1": make_scorer(lambda yt, yp: f1_score(yt, yp, average='macro')),
                    "accuracy": make_scorer(accuracy_score)}
    )
    cv_res = {key: mean(cv_res[key]) for key in cv_res}
    print(cv_res)


if __name__ == "__main__":
    multinomial_baseline()