from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import pandas as pd
from data_handler import DataHandler
from model_trainer import ModelTrainer
from utils import process_data, generate_map, generate_interactive_plots
from model import train_model, predict
from sklearn.metrics import mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.secret_key = 'sua-chave-secreta-única-e-segura'

data = None
model = None

# Função de validação do CSV
def validate_csv(file):
    try:
        df = pd.read_csv(file)
    except Exception as e:
        flash(f"Erro ao carregar o arquivo CSV: {e}", "error")
        return None

    expected_columns = ['sq_mt_built', 'sq_mt_useful', 'n_rooms', 'n_bathrooms', 'has_parking', 'buy_price']
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        flash(f"As seguintes colunas estão faltando: {', '.join(missing_columns)}", "error")
        return None

    return df


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def upload_file():
    global data
    if 'file' not in request.files:
        flash("Nenhum arquivo selecionado", "error")
        return redirect(request.url)

    file = request.files['file']
    if file and file.filename.endswith('.csv'):
        df = validate_csv(file)
        if df is None:
            return redirect(request.url)

        data = df
        statistics = data.describe(include='all').to_dict()
        flash("Arquivo CSV carregado e validado com sucesso!", "success")
        return render_template('index.html', statistics=statistics)

    flash("Por favor, envie um arquivo CSV válido.", "error")
    return redirect(url_for('index'))

@app.route('/analyze')
def analyze():
    global data
    if data is not None:
        stats, graphs = process_data(data)

        graphs = {}

        map_path, map_error = generate_map(data)
        if map_error:
            flash(map_error, "error")
        else:
            graphs['Mapa de Imóveis'] = map_path

        # Gerar gráficos interativos
        interactive_plots = generate_interactive_plots(data)
        if interactive_plots:

            graphs.update(interactive_plots)

        else:
            flash("Não foi possível gerar os gráficos interativos.", "error")

        graphs = {k: v for k, v in graphs.items() if v}

        return render_template('analyze.html', stats=stats, graphs=graphs)
    return redirect(url_for('index'))


@app.route('/train', methods=['GET', 'POST'])
def train():
    global data, model
    model_info = None
    model_trained = False

    if request.method == 'POST' and data is not None:
        classifier_type = request.form.get('classifier')
        max_depth = request.form.get('max_depth')
        n_neighbors = request.form.get('n_neighbors')

        model_info = train_model(data, classifier_type, max_depth, n_neighbors)
        model = model_info['model']
        model_trained = True

    return render_template('train.html', model_info=model_info, model_trained=model_trained)


@app.route('/predict', methods=['GET', 'POST'])
def make_prediction():
    global model
    prediction = None
    if request.method == 'POST':
        if model is None:
            try:
                model = joblib.load('model.pkl')
            except Exception as e:
                flash(f"Erro ao carregar o modelo: {e}", "error")
                return redirect(url_for('index'))

        user_input = [float(request.form[key]) for key in
                      ['sq_mt_built', 'sq_mt_useful', 'n_rooms', 'n_bathrooms', 'has_parking']]

        prediction = predict(user_input, model)

    return render_template('predict.html', prediction=prediction)


@app.route('/diagnose')
def diagnose():
    global data
    if data is not None:
        if 'latitude' not in data.columns or 'longitude' not in data.columns:
            flash("As colunas 'latitude' e 'longitude' não estão presentes no conjunto de dados.", "error")
            return render_template('diagnose.html')

        lat_lon_missing = data[['latitude', 'longitude']].isnull().sum()
        flash(f"Número de valores ausentes em 'latitude': {lat_lon_missing['latitude']}", "info")
        flash(f"Número de valores ausentes em 'longitude': {lat_lon_missing['longitude']}", "info")
        return render_template('diagnose.html', missing_data=lat_lon_missing.to_dict())
    else:
        flash("Nenhum dado disponível para diagnóstico.", "error")
        return redirect(url_for('index'))

data_handler = DataHandler('data/houses_Madrid.csv')

@app.route('/upload-new-data', methods=['POST'])
def upload_new_data():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'Nenhum arquivo foi enviado'}), 400

    new_data = pd.read_csv(file)

    data_handler.add_new_data(new_data)
    data_handler.save_updated_data()

    return jsonify({'message': 'Dados adicionados com sucesso!'})

from datetime import datetime

@app.route('/retrain', methods=['POST'])
def retrain_model():
    try:
        data = pd.read_csv('data/houses_Madrid.csv')

        if os.path.exists('model_vectorizer.pkl'):
            vectorizer = joblib.load('model_vectorizer.pkl')
        else:
            vectorizer = None

        trainer = ModelTrainer(data, vectorizer=vectorizer)

        model = trainer.train_model()

        trainer.save_model('model.pkl')

        return jsonify({"message": "Modelo re-treinado com sucesso!"})
    except Exception as e:
        return jsonify({"message": f"Erro ao re-treinar o modelo: {str(e)}"})

if __name__ == '__main__':
    app.run(debug=True)
