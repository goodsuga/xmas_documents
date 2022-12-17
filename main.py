from pathlib import Path
from utils import (
    OptimizableModelBase, make_predict_dataset, make_train_dataset
)
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from sklearn.pipeline import make_pipeline
import optuna
from optuna import visualization
import numpy as np
from tqdm import tqdm
import plotly.graph_objects as go
import numpy as np
from collections import Counter
from random import choices, randint
from copy import deepcopy
from sklearn.linear_model import LogisticRegression, SGDClassifier

class PhraseInterpreter:
    def __init__(self):
        pass

    def interpret_document(self, model, text, classnames, n_runs=200, features=5, document_name="", avg_confidence=None):
        base = model.predict_proba([text])[0]

        words = np.array(text.split(" "))
        imps = np.zeros((len(words), len(base)))

        for i in tqdm(list(range(n_runs))):
            take = np.random.choice(np.arange(0, len(words)), size=randint(1, len(words)-1), replace=False)
            pred = model.predict_proba([" ".join(words[take])])[0]
            diff = pred-base # полож. знач ==> важно для класса
            imps[take] += diff
        
        predicted_class = np.argmax(base)
        most_important = np.argsort(imps[:, predicted_class])
        texts = []
        for i in range(features):
            texts.append(' '.join(words[max(0, most_important[i]-10):min(len(words), most_important[i]+10)]))
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                r=base,
                theta=classnames,
                fill='toself',
                name="Уверенность модели",
            )
        )
        if avg_confidence is not None:
            fig.add_trace(
                 go.Scatterpolar(
                    r=avg_confidence,
                    theta=classnames,
                    fill='toself',
                    name="Средняя уверенность",
                    fillcolor="grey",
                    opacity=0.65
            )
        )
        fig.update_layout(font=dict(size=16))
        fig.show()
        return fig, texts

MODEL_BASES = {
    "logistic": OptimizableModelBase(
            LogisticRegression,
            {
                "solver": lambda trial: trial.suggest_categorical("solver", ["newton-cg", "sag", "saga", "lbfgs"]),
                "C": lambda trial: trial.suggest_float("C", 0.001, 10.0)
            }
        ),
    "sgd": OptimizableModelBase(
            SGDClassifier,
            {
                "loss": lambda trial: trial.suggest_categorical("loss", ['log_loss', 'modified_huber']),
                "penalty": lambda trial: trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
            }
    )
}

class DocumentClassifier:
    def __init__(self):
        pass

    def train(self, data_dir: Path, class_file: Path, allowed_models="all", max_iters=3):
        train_df = make_train_dataset(data_dir, class_file)
        class_map = {
            document_class: document_class_index
            for document_class_index, document_class 
            in zip(range(len(train_df['Класс документа'])), train_df['Класс документа'].unique())
        }
        print(class_map)
        self.class_map = class_map

        train_df['Класс документа (индекс)'] = train_df['Класс документа'].apply(class_map.get)

        self.cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=5,
            random_state=42
        )
        
        self.model_studies = {}
        for model in MODEL_BASES:
            obj = MODEL_BASES[model].get_optimization_objective(
                train_df["Текст документа"],
                train_df["Класс документа (индекс)"],
                self.cv
            )
            self.model_studies[model] = optuna.create_study(direction='maximize')
            self.model_studies[model].optimize(obj, n_trials=max_iters)

        self.models = {}
        self.info = {}
        self.best_model = None
        self.best_model_name = None
        best_score = None

        for modelname in MODEL_BASES:
            model = make_pipeline(
                MODEL_BASES[modelname].rebuild_vectorizer(self.model_studies[modelname].best_params),
                MODEL_BASES[modelname].rebuild_model(self.model_studies[modelname].best_params)
            )
            self.models[modelname] = model
            self.models[modelname].fit(
                train_df["Текст документа"],
                train_df["Класс документа (индекс)"]
            )
            if self.best_model is None:
                self.best_model = self.models[modelname]
                best_score = self.model_studies[modelname].best_value
                self.best_model_name = modelname
            else:
                if best_score < self.model_studies[modelname].best_value:
                    self.best_model = self.models[modelname]
                    best_score = self.model_studies[modelname].best_value
                    self.best_model_name = modelname

            self.info[modelname] = {
                "История оптимизации": optuna.visualization.plot_optimization_history(self.model_studies[modelname]),
                "Важность параметров": optuna.visualization.plot_param_importances(self.model_studies[modelname]),
                "F1(macro) на кросс-валидации": self.model_studies[modelname].best_value
            }
            self.info[modelname]['История оптимизации'].update_layout(
                title=f"История оптимизации {modelname}",
                title_x=0.5,
                yaxis_title="Значение метрики",
                xaxis_title="Шаг оптимизации"
            )
            self.info[modelname]['Важность параметров'].update_layout(
                title=f"Важность параметров {modelname}",
                title_x=0.5,
                yaxis_title="Гиперпараметр",
                xaxis_title="Важность для метрики"
            )

        return self.info

    def _transform_confidence(self, conf, avg_conf):
        xs = []
        ys = []
        ys_avg = []
        for classname, conf, avg_conf_val in zip(map(lambda key: key.replace("Договоры для акселератора/", ""),
                                                     self.class_map.keys()), conf[0], avg_conf[0]):
            xs.append(classname)
            ys.append(conf)
            ys_avg.append(avg_conf_val)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=xs,
            y=ys,
            name="Уверенность лучшей модели",
            opacity=0.9
        ))
        fig.add_trace(
            go.Bar(x=xs, y=ys_avg, name="Средняя уверенность", opacity=0.9)
        )
        fig.update_layout(title="Уверенность лучшей модели и средняя уверенность", title_x=0.5, font=dict(size=16))
        return fig

    def _avg_pred(self, text: str, mode='single'):
        pred = np.zeros((1, len(list(self.class_map.keys()))))
        count = 0
        for modelname in self.models:
            pred += self.models[modelname].predict_proba([text])
            count += 1
        pred /= count
        if mode == 'single':
            return np.argmax(pred)
        else:
            return pred
    
    def predict(self, data_dir: Path):
        df = make_predict_dataset(data_dir)
        reverse_class_map = {
            index: classname
            for index, classname
            in zip(self.class_map.values(), self.class_map.keys())
        }
        prediction_data = []
        for text in tqdm(df['Текст документа']):
            info = {
                "Прогноз (по лучшей модели)": reverse_class_map.get(self.best_model.predict([text])[0]),
                "Прогноз всех моделей (методом среднего)": reverse_class_map.get(self._avg_pred(text, 'single')),
                "Уверенность": self._transform_confidence(self.best_model.predict_proba([text]), self._avg_pred(text, 'all'))
            }
            info["Уверенность"].update_layout(
                title=f"Уверенность лучшей модели и средняя уверенность",
                title_x=0.5
            )
            info["Уверенность"].show()
            interpret_plot, texts = PhraseInterpreter().interpret_document(
                self.best_model, text, list(map(lambda key: key.replace("Договоры для акселератора/", ""), self.class_map.keys())),
                avg_confidence=self._avg_pred(text, 'all')[0]
            )
            info["Уверенность (лепестки)"] = interpret_plot
            info["Главные фразы"] = texts
            print(texts)
            interpret_plot.show()
            pprint(info)
            prediction_data.append(info)

        return prediction_data

from pprint import pprint
e = DocumentClassifier()
e.train(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs"),
        Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\classes.json"))

e.predict(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs_test"))
