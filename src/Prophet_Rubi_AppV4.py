#!/usr/bin/env python
# coding: utf-8

# In[90]:


#System settings
import sys
import warnings
warnings.filterwarnings("ignore")


# In[91]:


#Data Process and Model training
import pandas as pd

from neuralprophet import NeuralProphet, set_log_level
from neuralprophet import set_random_seed
set_random_seed(0)

from ipywidgets import *

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

from datetime import *
import plotly.express as px
import matplotlib.pyplot as plt

import numpy as np
from math import sqrt

import pickle

# In[92]:


#Connection PI
import PIconnect as PI
import xlwings as xw
import clr
from System.Net import NetworkCredential
from PIconnect.PIConsts import AuthenticationMode
from PIconnect.PIConsts import UpdateMode, BufferMode


# ### DESCARGA Y ACTUALIZACIÓN DE DATOS DESDE EL PI

# In[93]:


#DESCARGA DE DATOS

def filterarr(arr1):
    arr2=[]

    for e in arr1:
        try:
            arr2.append(float(e))
        except:    
            arr2.append(None)
    return arr2

def actualizacion_horario():
    
    #----------------------------------------HORARIO---------------------------------------------

    #Filtrado de casos repetidos
    df_actual = pd.read_csv("./input/Datos_Rubi_Pot.csv")
    df_actual = df_actual.drop_duplicates(subset=['Fecha'])
    df_actual.to_csv("./input/Datos_Rubi_Pot.csv",index=False)

    #Nombre del archivo a modificar
    df_actual = pd.read_csv("./input/Datos_Rubi_Pot.csv")

    #Extrae la fecha de últimos valores
    last_date = df_actual["Fecha"].values[-1]

    #Lista de Tags
    df_tag = pd.read_csv('./input/tag.txt', sep=",",header=None)
    lista = df_tag.values[0].tolist()
    
    #Fechas de inicio y final
    d_start = last_date
    d_end = datetime.now().strftime("%Y-%m-%d %H:00:00")

    #Proceso de descarga de datos
    if(d_start == d_end):
        print(">> Los datos horario ya están actualizados")
    else:   

        PI.PIConfig.DEFAULT_TIMEZONE = 'Etc/GMT+5'
        print(datetime.today())

        df = pd.DataFrame()

        print(PI.PIConfig.DEFAULT_TIMEZONE)

        # try:

        with PI.PIServer(server='PEREDGMOY1001', authentication_mode=AuthenticationMode.WINDOWS_AUTHENTICATION) as server:
    
            for e in lista:
                    print(e)
                    points = server.search(e)
                    data = points[0].interpolated_values(d_start, d_end, '1h')
                    print(points[0])

                    df["Time"]=data.index
                    df[str(e)]=filterarr(data.values)

        df["Time"] = df["Time"].dt.tz_localize(None)
        df['Time'] = df["Time"].dt.strftime("%d/%b/%Y %H:%M:%S")
        df['Time'] = pd.to_datetime(df['Time'],format="%d/%b/%Y %H:%M:%S")

        print(df)

        df[1:].to_csv('./input/Datos_Rubi_Pot.csv', mode='a', header=False, index = False)
        print(">> Potencia horaria actualizada")
                    
        # except Exception as e:
            
        #     print(">> Error 1: Falla en proceso de actualizacion horaria")
        #     print(">> Detalles del error:", e)
        #     txt = ">> Error 1: Falla en proceso de actualizacion horaria\n" + ">> Detalles del error: " + str(e)
        #     status(txt)

def actualizacion_minutal():
    
    #----------------------------------------15MINUTAL---------------------------------------------
    #Filtrado de casos repetidos
    df_actual = pd.read_csv("./input/Datos_Rubi_Pot_15m.csv")
    df_actual = df_actual.drop_duplicates(subset=['Fecha'])
    df_actual.to_csv("./input/Datos_Rubi_Pot_15m.csv",index=False)

    #Nombre del archivo a modificar
    df_actual = pd.read_csv("./input/Datos_Rubi_Pot_15m.csv")

    #Extrae la fecha de últimos valores
    last_date = df_actual["Fecha"].values[-1]

    #Lista de Tags
    df_tag = pd.read_csv('./input/tag.txt', sep=",",header=None)
    lista = df_tag.values[0].tolist()
    
    #Fechas de inicio y final
    d_start = last_date
    d_end = datetime.now().strftime("%Y-%m-%d %H:00:00")

    #Proceso de descarga de datos
    if(d_start == d_end):
        print(">> Los datos quinceminutal ya están actualizados")
    else:

        PI.PIConfig.DEFAULT_TIMEZONE = 'Etc/GMT+5'
        print(datetime.today())

        df = pd.DataFrame()

        print(PI.PIConfig.DEFAULT_TIMEZONE)

        # try:

        with PI.PIServer(server='PEREDGMOY1001', authentication_mode=AuthenticationMode.WINDOWS_AUTHENTICATION) as server:
    
            for e in lista:
                    print(e)
                    points = server.search(e)
                    data = points[0].interpolated_values(d_start, d_end, '15m')
                    print(points[0])

                    df["Time"]=data.index
                    df[str(e)]=filterarr(data.values)

        df["Time"] = df["Time"].dt.tz_localize(None)
        df['Time'] = df["Time"].dt.strftime("%d/%b/%Y %H:%M:%S")
        df['Time'] = pd.to_datetime(df['Time'],format="%d/%b/%Y %H:%M:%S")
        
        print(df)

        df[1:].to_csv('./input/Datos_Rubi_Pot_15m.csv', mode='a', header=False, index = False)
        print(">> Potencia quinceminutal actualizada")
                        
        # except Exception as e:
            
        #     print(">> Error 2: Falla en proceso de actualizacion quinceminutal")
        #     print(">> Detalles del error:", e)
        #     txt = ">> Error 2: Falla en proceso de actualizacion quinceminutal\n" + ">> Detalles del error: " + str(e)
        #     status(txt)
            


# ### CARGA DE DATOS AL PI

# In[94]:


#Carga de datos al servidor PI

def uploadPI(df_forecast,code):
    
    if code=='7DH':
        tag_name = "PES.Rubi.ForecastLH.Value"
    elif code=='2DH':
        tag_name = "PES.Rubi.ForecastSH.Value"    
    elif code=='7DM':
        tag_name = "PES.Rubi.ForecastLM.Value"    
    elif code=='2DM':
        tag_name = "PES.Rubi.ForecastSM.Value"
    
    #Fechas de inicio y final

    ind = df_forecast.loc[df_forecast['y'].isnull()].index[0]
    df_upload = df_forecast.loc[ind:][["ds","yhat1"]].copy()
    df_upload = df_upload.reset_index(drop=True, inplace=False)
    
    #Guardado de imagen de pronostico
    
    df_plot = pd.DataFrame()
    df_plot = df_upload.copy()
    df_plot.index = pd.to_datetime(df_plot["ds"])
    df_plot = df_plot.drop(["ds"],axis=1)

    df_plot.plot(title="Pronóstico "+str(code))
    plt.savefig("./output/"+str(code)+"/"+str(code)+"_pronostico.jpg")
    
    #Guardado de datos pronostico
    
    df_upload.to_csv("./output/"+str(code)+"/"+str(code)+"_cargadosPI.csv",index=False)

    d_start = df_upload['ds'].dt.strftime("%Y/%m/%d %H:%M:%S").values[0]
    d_end = df_upload['ds'].dt.strftime("%Y/%m/%d %H:%M:%S").values[-1]

    #Poceso de carga al PI
    
    PI.PIConfig.DEFAULT_TIMEZONE = 'Etc/GMT+5'

    print(PI.PIConfig.DEFAULT_TIMEZONE)
    
    try: 
        
        with PI.PIServer(server='PEREDGMOY1001', authentication_mode=AuthenticationMode.WINDOWS_AUTHENTICATION) as server:

            print(">> Cargando...")

            for i in range(len(df_upload['ds'].values)):
                points=server.search(tag_name)

                value = float(df_upload['yhat1'].tolist()[i])
                date = df_upload['ds'].dt.strftime("%Y/%m/%d %H:%M:%S").values[i]

                points[0].update_value(value,date,UpdateMode.REPLACE)
                data = points[0].recorded_values(d_start, d_end)

                print(date,str(value))

            print(">> Carga completa")

            print("\n>> Carga completa en el siguiente PIpoint: " + tag_name)
            
    except Exception as e:
            
            print(">> Error 3: Falla en proceso de carga al PI")
            print(">> Detalles del error:", e)
            txt = ">> Error 3: Falla en proceso de carga al PI\n" + ">> Detalles del error: " + str(e)
            status(txt)
        


# ### PROCESAMIENTO DE DATOS

# In[95]:


#Recuperacion datos vacios

def valuesNA(df_incomplete):
    
    df = df_incomplete.copy()
    cont = 1
    vacios = df.isnull().any(axis=1).sum()
    df['ds'] = pd.to_datetime(df['ds'])
    print(">> Recuperando datos de días atrás...")
    
    while(vacios>0):

        df_na = df[df.isnull().any(axis=1)].copy()
        
        print("Número de vacíos: " + str(vacios)+"\n")
        
        print("--------Registros Vacíos--------")
        print(df_na)
        print("\n")
        
        df_na['ds'] = pd.to_datetime(df_na['ds'])
        df_na_y = df_na.copy()
        
        df_na_y["ds"] = df_na_y["ds"] - timedelta(days = cont)
        
        cond_y = df['ds'].isin(df_na_y['ds'])
        values_y_pot = df.loc[cond_y, "y"]
        values_y_irr = df.loc[cond_y, "Irradiancia"]
        
        print("--------Registros Recuperados en base a días anteriores--------")
        print(df.loc[cond_y])
        print("\n")
        
        cond_t = df['ds'].isin(df_na['ds'])
        df.loc[cond_t,'y'] = values_y_pot.values
        df.loc[cond_t,'Irradiancia'] = values_y_irr.values

        cont+=1
        vacios = df.isnull().any(axis=1).sum()
        
        if(cont == 30):
            print("Se ha buscado hasta 30 días atrás, se recomienda llenar los datos manualmente")
            vacios = 0
        else:
            print(">> Recuperación exitosa!")
    
    return df


# In[96]:


#Filtrado de datos de X ultimos años para entrenamiento

def filter_data_train(df):
    
    df_tag = pd.read_csv('./input/Data_years_ago_to_train.txt', sep=",",header=None)
    years_ago = df_tag.values[0].tolist()[0]
    
    if(years_ago == "All"):
        
        df = df.copy()
    
    elif(int(years_ago)>0):
        
        df = df.copy()
        
        last_d = datetime.strptime(df["Fecha"][df.index[-1]],"%Y-%m-%d %H:%M:%S")
        first_d_dt = last_d - pd.DateOffset(years=years_ago)
        first_d = first_d_dt.strftime("%Y-%m-%d %H:%M:%S")
        first_d_index = df.loc[df["Fecha"] == first_d ].index[0]
        df = df.loc[df.index >= first_d_index]
        df = df.reset_index(drop=True)

    return df


# In[97]:


#Filtrado de datos de ultimo mes para prediccion

def filter_data_pred(df):
    
    df = df.copy()

    last_d = datetime.strptime(df["ds"][df.index[-1]],"%Y-%m-%d %H:%M:%S")
    first_d_dt = last_d - pd.DateOffset(days=8)
    first_d = first_d_dt.strftime("%Y-%m-%d %H:%M:%S")
    first_d_index = df.loc[df["ds"] == first_d ].index[0]
    df = df.loc[df.index >= first_d_index]
    df = df.reset_index(drop=True)

    return df


# In[98]:


#Pre procesamiento
def preP(code):
    print(code)
    
    try:
        
        if (code[-1:]=="H"):
            archive = "./input/Datos_Rubi_Pot.csv"
        elif(code[-1:]=="M"):
            archive = "./input/Datos_Rubi_Pot_15m.csv"

        df = pd.read_csv(archive)
        df = filter_data_train(df).copy()
        
        bu_columns = df.columns.tolist()
        df.columns = ['ds','y','Irradiancia']
        
        df.loc[df['y']<0,'y'] = 0
        df.loc[df['Irradiancia']<0,'Irradiancia'] = 0

        if(df.isnull().any(axis=1).sum()>0):
            df = valuesNA(df)
            df_save = df.copy()
            df_save.columns = bu_columns
            df_save.to_csv(archive, index=False)
        
        return df
    
    except Exception as e:
            
        print(">> Error 4: Falla en pre procesamiento de datos")
        print(">> Detalles del error:", e)
        txt = ">> Error 4: Falla en pre procesamiento de datos\n" + ">> Detalles del error: " + str(e)
        status(txt)
        


# In[99]:


#Post Procesamiento
def postP(df_forecast):
    
    try:
        
        df = df_forecast.copy()
        
        df_margin = df[["ds","yhat1"]].dropna()
        margin = float(df_margin.loc[((df_margin["ds"].dt.hour>=19)&(df_margin["ds"].dt.hour<=23)) | ((df_margin["ds"].dt.hour>=0)&(df_margin["ds"].dt.hour<6))]["yhat1"].mean())
        
        df["yhat1"] =  df["yhat1"] + (-1)*margin
        df.loc[(df["ds"].dt.hour>=19)&(df["ds"].dt.hour<=23),"yhat1"] = 0
        df.loc[(df["ds"].dt.hour>=0)&(df["ds"].dt.hour<6),"yhat1"] = 0
        df.loc[(df["yhat1"]<0),"yhat1"] = 0
        df.loc[(df["yhat1"]>151.8),"yhat1"] = 151.8
        
        return df
    
    except Exception as e:
            
        print(">> Error 5: Falla en post procesamiento de datos")
        print(">> Detalles del error:", e)
        txt = ">> Error 5: Falla en post procesamiento de datos\n" + ">> Detalles del error: " + str(e)
        status(txt)
    


# ### ENTRENAMIENTO Y PREDICCION

# In[100]:


def parameters(code):
    
    df_parameters = pd.read_csv('./models/'+str(code)+'/parameters.txt', sep=":",header=None)

    df_parameters.index = df_parameters[0]
    df_parameters = df_parameters.drop([0],axis=1)

    parameters = df_parameters.transpose().iloc[0].to_dict()
    
    parameters['ys'] = (parameters['ys'] == "True")
    parameters['ws'] = (parameters['ws'] == "True")
    parameters['ds'] = (parameters['ds'] == "True")
    
    return parameters


# In[101]:


#Reconocimiento de datos a utilizar según el modelo seleccionado

def model_parameters(code):
    
    try:
        parametros = {
                'n_lags':2,
                'n_forecasts':30,
                'ar_layers':[32,32],
                'learning_rate':0.001,
                'ys':True,
                'ws':True,
                'ds':True,
        }


        if(code == "2DH"):
            
            p = parameters(code)

            parametros = {
                'n_lags':int(p['days_lags'])*24,
                'n_forecasts':int(p['days_forecasts'])*24,
                'ar_layers':eval(p['ar_layers']),
                'learning_rate':float(p['learning_rate']),
                'ys':p['ys'],
                'ws':p['ws'],
                'ds':p['ds']
            }
            
            print(parametros)

        elif(code == "7DH"):
            
            p = parameters(code)

            parametros = {
                'n_lags':int(p['days_lags'])*24,
                'n_forecasts':int(p['days_forecasts'])*24,
                'ar_layers':eval(p['ar_layers']),
                'learning_rate':float(p['learning_rate']),
                'ys':p['ys'],
                'ws':p['ws'],
                'ds':p['ds']
            }
            
            print(parametros)

        elif(code == "2DM"):

            p = parameters(code)

            parametros = {
                'n_lags':int(p['days_lags'])*24*4,
                'n_forecasts':int(p['days_forecasts'])*24*4,
                'ar_layers':eval(p['ar_layers']),
                'learning_rate':float(p['learning_rate']),
                'ys':p['ys'],
                'ws':p['ws'],
                'ds':p['ds']
            }
            
            print(parametros)

        elif(code == "7DM"):

            p = parameters(code)

            parametros = {
                'n_lags':int(p['days_lags'])*24*4,
                'n_forecasts':int(p['days_forecasts'])*24*4,
                'ar_layers':eval(p['ar_layers']),
                'learning_rate':float(p['learning_rate']),
                'ys':p['ys'],
                'ws':p['ws'],
                'ds':p['ds']
            }
        
            print(parametros)
        
        else:
            print("Codigo de modelo no encontrado")

        return code,parametros
    
    except Exception as e:
            
        print(">> Error 6: Falla en seleccion de modelo")
        print(">> Detalles del error:", e)
        txt = ">> Error 6: Falla en seleccion de modelo\n" + ">> Detalles del error: " + str(e)
        status(txt)


# In[102]:


#Entrenamiento del modelo

def training(code, parametros):
    
    try:
        if (code[-1:]=="H"):
            frequency = "H"
        elif(code[-1:]=="M"):
            frequency = "15min"

        df = preP(code).copy()
        df["ds"] = pd.to_datetime(df["ds"])

        #Modelo

        m = NeuralProphet(
            n_lags=parametros['n_lags'],
            n_forecasts=parametros['n_forecasts'],
            #ar_layers=parametros['ar_layers'],
            num_hidden_layers=len(parametros['ar_layers']),
            d_hidden=parametros['ar_layers'][0],
            learning_rate=parametros['learning_rate'],
            yearly_seasonality=parametros['ys'],
            weekly_seasonality=parametros['ws'],
            daily_seasonality=parametros['ds'],
            optimizer='AdamW',
        )

        m.add_lagged_regressor("Irradiancia")

        df_train, df_test = m.split_df(df, freq=frequency, valid_p = 0.3)
        metrics = m.fit(df_train, freq=frequency, validation_df=df_test)
        #m = m.highlight_nth_step_ahead_of_each_forecast(parametros['n_forecasts'])

        #Guardado

        with open('./models/'+str(code)+'/'+str(code)+'.pkl','wb') as f:
            pickle.dump(m, f)
            print('>> Modelo '+str(code)+' entrenado y guardado en ./models ')
            
        #Guardado imagen
        
        fig = m.plot_parameters()
        fig.savefig('./models/'+ str(code) + '/'+ 'train.jpg')
        
        #Guardado de metricas
        with open("./models/"+str(code)+"/metrics.txt", "w") as archivo:
            archivo.write("Metricas del modelo:\n")
            archivo.write(metrics.tail(1).to_string())
    
    except Exception as e:
            
        print(">> Error 7: Falla en entrenamiento")
        print(">> Detalles del error:", e)
        txt = ">> Error 7: Falla en entrenamiento\n" + ">> Detalles del error: " + str(e)
        status(txt)


# In[107]:


#Prediccion filtrado de salida

def prediction(code):
    
    try:
    
        if (code[0]=="2"):
            d = 2
        elif(code[0]=="7"):
            d = 7

        df = preP(code).copy()
        df = filter_data_pred(df).copy()
        df['ds'] = pd.to_datetime(df['ds'])

        #Se carga modelo

        with open('./models/'+str(code)+'/'+str(code)+'.pkl','rb') as f:
            m = pickle.load(f)

        df_future = m.make_future_dataframe(df, n_historic_predictions=True)
        forecast = m.predict(df_future)

        cont=0
        df_forecast = pd.DataFrame()
        df_forecast = forecast[["ds","y","yhat1"]].copy()

        for i in range(len(forecast)):

            if pd.isnull(forecast["y"][i]):
                cont+=1
                df_forecast["yhat1"][i] = forecast["yhat"+str(cont)][i]

        df_forecast = postP(df_forecast).copy()
        
        #Guardado de imagen

        fig = m.plot(forecast)
        fig.savefig('./output/'+ str(code) + '/'+ 'model_behavior.jpg', bbox_inches='tight')
        
        df_forecast.index = df_forecast["ds"]
        df_forecast[["y","yhat1"]].plot()
        plt.savefig('./output/'+ str(code) + '/'+ 'model_behavior_post.jpg', bbox_inches='tight')

        return df_forecast
    
    except Exception as e:
        
        print(">> Error 8: Falla en prediccion")
        print(">> Detalles del error:", e)
        txt = ">> Error 8: Falla en prediccion\n" + ">> Detalles del error: " + str(e)
        status(txt)


# ### PRINCIPAL

# In[108]:


#Escribir en error en txt
def status(txt):
    
    with open("./output/status.txt", "w") as archivo:
        archivo.write(txt)


# In[109]:


def proccesses(argument):
    
    status("OK")
    
    actualizacion_horario()
    actualizacion_minutal()
        
    if (argument=="predict_2DH"):
        
        print("--------------2DH--------------")
        cod, parametros = model_parameters("2DH")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
    
    elif (argument=="predict_7DH"):
        
        print("--------------7DH--------------")
        cod, parametros = model_parameters("7DH")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
    
    elif (argument=="predict_2DM"):
        
        print("--------------2DM--------------")
        cod, parametros = model_parameters("2DM")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
    
    elif (argument=="predict_7DM"):
        
        print("--------------7DM--------------")
        cod, parametros = model_parameters("7DM")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
        
    elif (argument=="predict_all"):
        
        print("--------------2DH--------------")
        cod, parametros = model_parameters("2DH")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
        
        print("--------------7DH--------------")
        cod, parametros = model_parameters("7DH")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
        
        print("--------------2DM--------------")
        cod, parametros = model_parameters("2DM")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
        
        print("--------------7DM--------------")
        cod, parametros = model_parameters("7DM")
        df_forecast = prediction(cod)
        uploadPI(df_forecast,cod)
    
    elif (argument=="train_2DH"):
        
        print("--------------2DH--------------")
        cod, parametros = model_parameters("2DH")
        training(cod,parametros)
    
    elif (argument=="train_7DH"):
        
        print("--------------7DH--------------")
        cod, parametros = model_parameters("7DH")
        training(cod,parametros)
    
    elif (argument=="train_2DM"):
        
        print("--------------2DM--------------")
        cod, parametros = model_parameters("2DM")
        training(cod,parametros)
        
    elif (argument=="train_7DM"):
        
        print("--------------7DM--------------")
        cod, parametros = model_parameters("7DM")
        training(cod,parametros)
        
    elif (argument=="train_all"):
        
        print("--------------2DH--------------")
        cod, parametros = model_parameters("2DH")
        training(cod,parametros)
        print("--------------7DH--------------")
        cod, parametros = model_parameters("7DH")
        training(cod,parametros)
        print("--------------2DM--------------")
        cod, parametros = model_parameters("2DM")
        training(cod,parametros)
        print("--------------7DM--------------")
        cod, parametros = model_parameters("7DM")
        training(cod,parametros)
    
    
    
    else:
        print(">> Argumento no encontrado")
        


# In[110]:


if __name__ == "__main__":
    
    if len(sys.argv)>1:
        
        print(">> Argumento:",sys.argv[1])
        argument = sys.argv[1] 
    
    # try:
        
    proccesses(argument)
        
    # except Exception as e:
            
    #     print(">> Error encontrado\n")
    #     print(">> Detalles del error:", e)


# In[ ]:




