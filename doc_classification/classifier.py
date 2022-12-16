import optuna
from optuna import visualization
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from tqdm import tqdm
from pprint import pprint

from .utils import (
    make_train_dataset, configure_models, make_predict_dataset,
    _rebuild_sgd, _rebuild_logistic, _rebuild_vec,
    explain_instance, clear_texts
)




class DocumentClassifier:
    def __init__(self):
        pass

    def train(self, data_path: Path, allowed_models="all", max_iters=150):
        train_df = pd.read_parquet(data_path)
        train_df.loc[:, "Текст документа"] = clear_texts(train_df['Текст документа'])
        class_map = pd.factorize(train_df['Класс документа'])
        class_map = {
            document_class: document_class_index
            for document_class_index, document_class
            in zip(class_map[0], class_map[1])
        }
        self.class_map = class_map

        train_df['Класс документа (индекс)'] = train_df['Класс документа'].apply(class_map.get)
        print(train_df)
        allowed = [
           "linear", 
            "sgd"
        ]

        self.cv = RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=5,
            random_state=42
        )

        model_studies = configure_models(
            train_df["Текст документа"],
            train_df["Класс документа (индекс)"],
            self.cv,
            allowed,
            max_iters={"linear": 5, "sgd": 5}
        )
        self.model_studies = model_studies

        self.models = {}
        self.info = {}
        self.best_model = None
        self.best_model_name = None
        best_score = None

        for modelname in model_studies:
            vec = _rebuild_vec(model_studies[modelname].best_params)
            model_studies[modelname].best_params['cv'] = self.cv
            if modelname == "linear":
                model = _rebuild_logistic(model_studies[modelname].best_params)
            elif modelname == "sgd":
                model = _rebuild_sgd(model_studies[modelname].best_params)

            self.models[modelname] = make_pipeline(vec, model)
            self.models[modelname].fit(
                train_df["Текст документа"],
                train_df["Класс документа (индекс)"]
            )
            if self.best_model is None:
                self.best_model = self.models[modelname]
                best_score = model_studies[modelname].best_value
                self.best_model_name = modelname
            else:
                if best_score < model_studies[modelname].best_value:
                    self.best_model = self.models[modelname]
                    best_score = model_studies[modelname].best_value
                    self.best_model_name = modelname

            self.info[modelname] = {
                "История оптимизации": optuna.visualization.plot_optimization_history(model_studies[modelname]),
                "Важность параметров": optuna.visualization.plot_param_importances(model_studies[modelname]),
                "F1(macro) на кросс-валидации": model_studies[modelname].best_value
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

    def _transform_confidence(self, conf):
        xs = []
        ys = []
        for classname, conf in zip(self.class_map.keys(), conf[0]):
            xs.append(classname)
            ys.append(conf)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=xs,
            y=ys
        ))
        return fig

    def _avg_pred(self, text: str, mode='single'):
        pred = np.zeros((1, len(list(self.class_map.keys())) - 1))
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
                "Уверенность лучшей модели": self._transform_confidence(self.best_model.predict_proba([text])),
                "Средняя уверенность (по всем моделям)": self._transform_confidence(self._avg_pred(text, 'all')),
                "Прогноз всех моделей (методом среднего)": reverse_class_map.get(self._avg_pred(text, 'single')),
            }
            info["Уверенность лучшей модели"].update_layout(
                title=f"Уверенности {self.best_model_name}",
                title_x=0.5
            )
            info["Средняя уверенность (по всем моделям)"].update_layout(
                title=f"Средняя уверенность по всем моделям",
                title_x=0.5
            )
            info["Уверенность лучшей модели"].show()
            info["Средняя уверенность (по всем моделям)"].show()
            info["График уверенности и главных слов"] = {}
            for modelname in self.models:
                info["График уверенности и главных слов"][modelname] =\
                    explain_instance(self.models[modelname], self.class_map, text, document_name="Пример названия документа")
                #info["График уверенности и главных слов"][modelname].show()
            pprint(info)
            prediction_data.append(info)

        return prediction_data




if __name__ == "__main__":
    e = DocumentClassifier()
    e.train(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs"),
            Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\classes.json"))

    e.predict(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs_test"))