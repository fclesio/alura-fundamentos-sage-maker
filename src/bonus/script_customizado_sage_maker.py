from __future__ import print_function

import argparse
import os
import sys
import pandas as pd
import numpy as np
import sklearn
import json
from io import StringIO
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier

from sagemaker_containers.beta.framework import (
    content_types, encoders, env, modules, transformer, worker)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=100)
    
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])

    args = parser.parse_args()

    input_files = [ os.path.join(args.train, file) for file in os.listdir(args.train) ]
    if len(input_files) == 0:
        raise ValueError((f'Arquivos de treinamento não encontrados. Consultar o endereço {args.train}'))
    
    # Leitura dos dados no bucket do S3
    dados_treino = [pd.read_csv(file, header=None, engine="python") for file in input_files]
    
    # Se houver mais de 1 arquivo, vamos concatenar 
    # tudo em um unico dataframe 
    dados_treino = pd.concat(dados_treino)
 
    # Vamos pegar todas as linhas a contar da segunda
    # para o "dados_treino" e vamos definir as colunas
    # do dataframe
    dados_treino, dados_treino.columns = dados_treino[1:] , dados_treino.iloc[0]
    
    # Se quisermos, podemos converter colunas 
    # dentro do script de treinamento 
    dados_treino['CREDITO_CONCEDIDO'] = dados_treino['CREDITO_CONCEDIDO'].astype(int)

    # Dados de treinamento
    X_treino = dados_treino.iloc[:, 1:]
    y_treino = dados_treino.iloc[:, :-23]

    # Treinamento do modelo
    rfc = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, n_jobs=-1, random_state=42)
    rfc.fit(X_treino, y_treino.values.ravel())

    # Serializacao do modelo
    joblib.dump(rfc, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserializacao e retorno do modelo
    
    Deve ter o mesmo nome do modelo retornado no main
    """
    rfc = joblib.load(os.path.join(model_dir, "model.joblib"))
    return rfc


def input_fn(input_data, content_type):
    """Faz o parseamento dos dados de input. 
    
    Aceita somente o input do tipo csv para este modelo. 
    Aqui não fazemos nenhuma checagem se numero de colunas. Isso
    pode ser feito em outras etapas. 
    """
    if content_type == 'text/csv':
        df = pd.read_csv(StringIO(input_data), header=None)
        return df
    else:
        raise ValueError(f"O tipo {content_type} não é suportado pelo script!. Este script aceita somente o input text/csv.")

        
def predict_fn(input_data, model):
    """Faz a predição do nosso modelo"""
    prediction = model.predict(input_data)
    pred_proba = model.predict_proba(input_data)    
    return np.array([prediction, pred_proba[0]])


def output_fn(prediction, accept):
    """Formato do output da predição

    O formato de aceitação (accept) do tipo de conteudo (accept/content-type)
    entre os containers é JSON. Nos tambem queremos definir o ContentType
    ou o mimetype como o mesmo valor como o formato de aceitação então
    o próximo container pode ler a resposta do payload corretamente.
    """
    if accept == "application/json":
        instances = []
        for row in prediction.tolist():
            instances.append({"features": row})

        json_output = {"instances": instances}
        json_output = pd.Series(json_output).to_json(orient='values')

        return worker.Response(json.dumps(json_output), mimetype=accept)
    elif accept == 'text/csv':
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
    else:
        return worker.Response(encoders.encode(prediction, accept), mimetype=accept)
