from pathlib import Path
from utils import (
    make_train_dataset, configure_models, make_predict_dataset,
    _rebuild_sgd, _rebuild_logistic, _rebuild_vec
)
from sklearn.model_selection import RepeatedStratifiedKFold
import pandas as pd
from sklearn.pipeline import make_pipeline
import optuna
from optuna import visualization

class DocumentClassifier:
    def __init__(self):
        pass

    def train(self, data_dir: Path, class_file: Path, allowed_models="all", max_iters=150):
        train_df = make_train_dataset(data_dir, class_file)
        class_map = pd.factorize(train_df['Класс документа'])
        class_map = {
            document_class: document_class_index
            for document_class_index, document_class 
            in zip(class_map[0], class_map[1])
        }
        self.class_map = class_map

        train_df['Класс документа (индекс)'] = train_df['Класс документа'].apply(class_map.get)
        allowed = ["linear", "sgd"]

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
        best_score = None

        for modelname in model_studies:
            vec = _rebuild_vec(model_studies[modelname].best_params)
            model_studies[modelname].best_params['cv'] = self.cv
            if modelname == "linear":
                model = _rebuild_logistic(model_studies[modelname].best_params)
            elif modelname == "sgd":
                model = _rebuild_sgd(model_studies[modelname].best_params)
            
            self.models[modelname] = make_pipeline(vec, model)
            if self.best_model is None:
                self.best_model = self.models[modelname]
                best_score = model_studies[modelname].best_value
            else:
                if best_score < model_studies[modelname].best_value:
                    self.best_model = self.models[modelname]
                    best_score = model_studies[modelname].best_value

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

    def predict(self, data_dir: Path):
        df = make_predict_dataset(data_dir)
        prediction_data = []
        for text in make_predict_dataset['Текст документа']:

        return self.best_model.predict(df["Текст документа"])


e = DocumentClassifier()
e.train(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs"),
        Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\classes.json"))
