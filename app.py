import os
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, redirect

from doc_classification.classifier import DocumentClassifier




app = Flask(__name__)
clf = DocumentClassifier()


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template("index.html")


@app.route('/train', methods=['GET', 'POST'])
def train():

    app.config['UPLOAD_FOLDER'] = './train_files'
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

    if request.method == 'POST':

        # get parquet file
        f = request.files['file']
        filename = secure_filename(f.filename)
        train_df = os.path.join(app.config['UPLOAD_FOLDER'], 'train.pqt')
        f.save(train_df)

        train_info = clf.train(Path(train_df))
        print(train_info)

        Path('./static/images').mkdir(parents=True, exist_ok=True)
        graph_paths = []
        metrics = []
        for model, graph_dict in train_info.items():

            graph_path = f'История оптимизации {model}'
            graph_dict["История оптимизации"].write_image(os.path.join('./static/images', f'{graph_path}.jpg'))
            graph_paths.append(graph_path)

            graph_path = f'Важность параметров {model}'
            graph_dict["Важность параметров"].write_image(os.path.join('./static/images', f'{graph_path}.jpg'))
            graph_paths.append(graph_path)

            metrics.append({
                'name': f'F1(macro) на кросс-валидации {model}',
                'value': graph_dict["F1(macro) на кросс-валидации"]
            })

        return render_template("train_results.html", img_paths=graph_paths, metrics=metrics)

    return render_template("train.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    app.config['UPLOAD_FOLDER'] = './predict_files'
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)

    if request.method == 'POST':

        # get parquet file
        f = request.files['file']
        filename = secure_filename(f.filename)
        df = os.path.join(app.config['UPLOAD_FOLDER'], 'test.pqt')
        f.save(df)

        data = pd.read_parquet(df)
        texts = data['Текст документа'].tolist()
        idxs = np.arange(len(texts))

        preds = clf.predict(Path(df))
        print(preds)

        Path('./static/images').mkdir(parents=True, exist_ok=True)
        preds_info = []

        for ix, doc in enumerate(preds):

            graph_path1 = f"Уверенность_док_{ix}"
            doc["Уверенность"].write_image(os.path.join('./static/images', f'{graph_path1}.jpg'))

            graph_path2 = f"Уверенность (лепестки)_док_{ix}"
            doc["Уверенность (лепестки)"].write_image(os.path.join('./static/images', f'{graph_path2}.jpg'))

            doc_info = {
                "graph1": graph_path1,
                "graph2": graph_path2,
                "text": doc["Главные фразы"]
            }
            preds_info.append(doc_info)


        return render_template("predict_results.html", preds_info=preds_info)

    return render_template("predict.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
   return render_template('upload.html')




if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
