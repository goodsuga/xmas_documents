import os
import zipfile
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, url_for, request, redirect

from doc_classification.classifier import DocumentClassifier

app = Flask(__name__)

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

        clf = DocumentClassifier()
        train_info = clf.train(Path(train_df))
        print(train_info)

        Path('./static/images').mkdir(parents=True, exist_ok=True)
        graph_paths = []
        for model, graph_dict in train_info.items():
            graph_path = f'История_оптимизации_{model}'
            graph_dict["История оптимизации"].write_image(os.path.join('./static/images', graph_path))
            graph_paths.append(graph_path)
            graph_path = f'Важность_параметров_{model}'
            graph_dict["Важность параметров"].write_image(os.path.join('./static/images', graph_path))
            graph_paths.append(graph_path)

        return render_template("train_results.html", img_paths=graph_paths)

    return render_template("train.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    app.config['UPLOAD_FOLDER'] = './predict_files'
    Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
    if request.method == 'POST':
        f = request.files['file']
        filename = secure_filename(f.filename)
        f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        clf.predict(Path(r"C:\Users\teberda\Documents\GitHub\xmas_documents\docs_test"))

    return render_template("predict.html")


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
   return render_template('upload.html')





if __name__ == "__main__":
    app.run(debug=True)
