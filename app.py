import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
from PIL import Image
import base64
import io
import pickle
from keras.models import load_model

def predict_malaria(img):
    try:
        img = img.resize((36, 36))
        img = np.asarray(img)
        img = img.reshape((1, 36, 36, 3))
        img = img.astype(np.float64)
        model = load_model("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/malaria.h5")
        pred_probs = model.predict(img)[0]
        pred_class = np.argmax(pred_probs)
        pred_prob = pred_probs[pred_class]
        return pred_class, pred_prob
    except Exception as e:
        print(f"Erro na predição da malária: {e}")
        return None, None

def predict_pneumonia(img):
    try:
        img = img.convert('L')
        img = img.resize((36, 36))
        img = np.asarray(img)
        img = img.reshape((1, 36, 36, 1))
        img = img / 255.0
        model = load_model("/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/pneumonia.h5")
        pred_probs = model.predict(img)[0]
        pred_class = np.argmax(pred_probs)
        pred_prob = pred_probs[pred_class]
        return pred_class, pred_prob
    except Exception as e:
        print(f"Erro na predição de pneumonia: {e}")
        return None, None

def predict_diabetes(user_input):
    try:
        user_input = [float(x) for x in user_input]
        with open('/Users/thanizeassuncaorodrigues/Documents/GitHub/DiagnoSys/models/diabetes_model.sav', 'rb') as file:
            diabetes_model = pickle.load(file)
        diab_prediction = diabetes_model.predict([user_input])
        return 'A pessoa é diabética' if diab_prediction[0] == 1 else 'A pessoa não é diabética'
    except Exception as e:
        print(f"Erro na predição de diabetes: {e}")
        return 'Erro na predição de diabetes'

# Initialize the Dash app
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='home', children=[
        dcc.Tab(label='Página Inicial', value='home'),
        dcc.Tab(label='Detecção Malária', value='malaria'),
        dcc.Tab(label='Detecção Pneumonia', value='pneumonia'),
        dcc.Tab(label='Detecção Diabetes', value='diabetes'),
        dcc.Tab(label='Datasets Disponíveis', value='datasets')
    ]),
    html.Div(id='tab-content')
])

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value')]
)
def update_content(tab_name):
    if tab_name == 'home':
        return html.Div([
            html.H1('Bem-vindo à Aplicação de Previsão de Anomalias Médicas'),
            html.P("Este é um projeto de previsão de diversas anomalias médicas usando modelos de deep learning e machine learning."),
            html.P("É importante observar que os modelos utilizados nesta aplicação foram obtidos de repositórios públicos na internet e, portanto, sua confiabilidade pode variar."),
            html.P("Embora tenham sido treinados em grandes conjuntos de dados médicos, é fundamental lembrar que todas as previsões devem ser verificadas por profissionais de saúde qualificados."),
            html.H2("Perguntas Frequentes"),
            html.Div([
                html.Div([
                    html.H3("Como a previsão de anomalias é feita?"),
                    html.P("A detecção de pneumonia e malária é feita usando uma rede neural convolucional (CNN), enquanto a seção de diabetes é detectada por um modelo Random Forest.")
                ]),
                html.Div([
                    html.H3("Os modelos são precisos?"),
                    html.P("Os modelos foram treinados em grandes conjuntos de dados médicos, mas lembre-se de que todas as previsões devem ser verificadas por profissionais de saúde qualificados.")
                ]),
                html.Div([
                    html.H3("Qual é o propósito desta aplicação?"),
                    html.P("Esta aplicação foi desenvolvida para auxiliar na detecção de diversas anomalias médicas em imagens de diferentes partes do corpo.")
                ]),
                html.Div([
                    html.H3("Quais tipos de anomalias médicas podem ser detectadas?"),
                    html.P("Os modelos podem detectar várias anomalias, incluindo pneumonia, malária e diabetes.")
                ]),
            ]),
        ])
    
    elif tab_name == 'malaria':
        return html.Div([
            html.H1('Previsão de Malária'),
            dcc.Upload(
                id='upload-image-malaria',
                children=html.Button('Faça o upload de uma imagem'),
                multiple=False
            ),
            html.Div(id='malaria-image'),
            html.Div(id='malaria-result')
        ])
    
    elif tab_name == 'pneumonia':
        return html.Div([
            html.H1('Previsão de Pneumonia'),
            dcc.Upload(
                id='upload-image-pneumonia',
                children=html.Button('Faça o upload de uma imagem'),
                multiple=False
            ),
            html.Div(id='pneumonia-image'),
            html.Div(id='pneumonia-result')
        ])
    
    elif tab_name == 'diabetes':
        return html.Div([
            html.H1('Previsão de Diabetes'),
            dcc.Input(id='pregnancies', type='number', placeholder='Número de Gestações'),
            dcc.Input(id='glucose', type='number', placeholder='Nível de Glicose'),
            dcc.Input(id='bloodpressure', type='number', placeholder='Valor da Pressão Arterial'),
            dcc.Input(id='skinthickness', type='number', placeholder='Valor da Espessura da Pele'),
            dcc.Input(id='insulin', type='number', placeholder='Nível de Insulina'),
            dcc.Input(id='bmi', type='number', placeholder='Valor do IMC'),
            dcc.Input(id='diabetespedigree', type='number', placeholder='Valor da Função de Pedigree de Diabetes'),
            dcc.Input(id='age', type='number', placeholder='Idade da Pessoa'),
            html.Button('Resultado do Teste de Diabetes', id='diabetes-button'),
            html.Div(id='diabetes-result')
        ])
    
    elif tab_name == 'datasets':
        return html.Div([
            html.H1('Datasets Disponíveis'),
            html.P("Esta página contém links para download e visualização de datasets utilizados na aplicação."),
            html.Div([
                html.Div([
                    html.H3("Dataset de Malária"),
                    html.A("Download", href="https://www.kaggle.com/datasets/iarunava/cell-images-for-detecting-malaria")
                ]),
                html.Div([
                    html.H3("Dataset de Pneumonia"),
                    html.A("Download", href="https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
                ]),
                html.Div([
                    html.H3("Dataset de Doenças Cardíacas"),
                    html.A("Download", href="https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/heart.csv")
                ]),
                html.Div([
                    html.H3("Dataset de Doenças Renais"),
                    html.A("Download", href="https://www.kaggle.com/datasets/mansoordaku/ckdisease")
                ]),
                html.Div([
                    html.H3("Dataset de Diabetes"),
                    html.A("Download", href="https://github.com/siddhardhan23/multiple-disease-prediction-streamlit-app/blob/main/dataset/diabetes.csv")
                ]),
                html.Div([
                    html.H3("Dataset de Doenças Hepáticas"),
                    html.A("Download", href="https://www.kaggle.com/datasets/uciml/indian-liver-patient-records")
                ]),
                html.Div([
                    html.H3("Dataset de Câncer de Mama"),
                    html.A("Download", href="https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data")
                ])
            ])
        ])

@app.callback(
    Output('malaria-image', 'children'),
    [Input('upload-image-malaria', 'contents')]
)
def update_malaria_image(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        return html.Div([
            html.H3('Imagem Carregada:'),
            html.Img(src=contents, style={'width': '50%'}),
        ])
    return 'Faça o upload de uma imagem para visualizar'

@app.callback(
    Output('pneumonia-image', 'children'),
    [Input('upload-image-pneumonia', 'contents')]
)
def update_pneumonia_image(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        return html.Div([
            html.H3('Imagem Carregada:'),
            html.Img(src=contents, style={'width': '50%'}),
        ])
    return 'Faça o upload de uma imagem para visualizar'

@app.callback(
    Output('malaria-result', 'children'),
    [Input('upload-image-malaria', 'contents')]
)
def update_malaria_result(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        pred_class, pred_prob = predict_malaria(image)
        if pred_class is not None:
            return f"Previsão: {'Malária' if pred_class == 1 else 'Sem Malária'}, Probabilidade: {pred_prob:.2f}"
        else:
            return 'Erro na previsão de malária'
    return 'Faça o upload de uma imagem para ver o resultado'

@app.callback(
    Output('pneumonia-result', 'children'),
    [Input('upload-image-pneumonia', 'contents')]
)
def update_pneumonia_result(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        image = Image.open(io.BytesIO(decoded))
        pred_class, pred_prob = predict_pneumonia(image)
        if pred_class is not None:
            return f"Previsão: {'Pneumonia' if pred_class == 1 else 'Sem Pneumonia'}, Probabilidade: {pred_prob:.2f}"
        else:
            return 'Erro na previsão de pneumonia'
    return 'Faça o upload de uma imagem para ver o resultado'

@app.callback(
    Output('diabetes-result', 'children'),
    [Input('diabetes-button', 'n_clicks')],
    [Input('pregnancies', 'value'),
     Input('glucose', 'value'),
     Input('bloodpressure', 'value'),
     Input('skinthickness', 'value'),
     Input('insulin', 'value'),
     Input('bmi', 'value'),
     Input('diabetespedigree', 'value'),
     Input('age', 'value')]
)
def update_diabetes_result(n_clicks, pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age):
    if n_clicks is not None and None not in [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]:
        user_input = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigree, age]
        result = predict_diabetes(user_input)
        return result
    return 'Por favor, preencha todos os campos e clique no botão para obter o resultado.'

if __name__ == '__main__':
    app.run_server(debug=True)
