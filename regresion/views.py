# Create your views here.
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import io
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from regresion_lineal import settings
import os
@csrf_exempt
def csv_to_html(request):
    
    ruta_imagen = os.path.join(settings.MEDIA_ROOT, 'mi_imagen.png')
    if os.path.isfile(ruta_imagen):
        os.remove(ruta_imagen)
            
    data= pd.read_csv('data/finaldata.csv') # lee el archivo CSV
    top_5 = data.head(10)
    
    data_out = top_5.to_html() # convierte los datos en HTML
    
    # Crea un gráfico utilizando la función regplot de seaborn
    grafico = sns.regplot(x=data['year'], y=data['count_zw'], data=data, order = 0,ci = None, scatter_kws={'color': 'r'})

    # Convierte el gráfico en una imagen PNG
    buffer = io.BytesIO()
    plt.savefig(ruta_imagen, format='png')
    buffer.seek(0)
    return render(request, 'csv_to_html.html', {'data': data_out, 'image_regresion': ruta_imagen, 'datos': regresion_lineal(request)})

def regresion_lineal(request):
     
    test = 2025
    try:
        if request.method == 'POST':
            form = request.POST.dict()
            if form['valor_anio']:
                test = int(form['valor_anio'])
    except Exception as e:
        print(e)
    df = pd.read_csv('data/finaldata.csv') # lee el archivo CSV
    model = LinearRegression(fit_intercept=True)
    feature_cols = ['year']
    valor_x = df[feature_cols]
    valor_y = df.count_zw
    print('Shape X:', valor_x.shape)
    print('Type X:', type(valor_x))
    print('Shape y:', valor_y.shape)
    print('Type y:', type(valor_y))
    Xtrain, Xtest, ytrain, ytest = train_test_split(valor_x, valor_y, random_state = 1)
    model.fit(Xtrain, ytrain)
    test_sklearn = np.array(test).reshape(-1,1) # .reshape(-1,1) solo para regresión univariable
    ypred = model.predict(Xtest)
    print("ANIO:", test)
    return {
        'anio': test,
        'prediccion': model.predict(test_sklearn),
        'MAE': mean_absolute_error(ytest, ypred).round(2),
        'MSE': mean_squared_error(ytest, ypred).round(2),
        'RMSE': np.sqrt(mean_squared_error(ytest, ypred)).round(2),
        'R2': r2_score(ytest, ypred).round(2)
    }