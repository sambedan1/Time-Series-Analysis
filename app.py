import plotly
from flask import Flask, render_template,url_for,flash,redirect,request,session
import sklearn
import pickle
from flask import Flask, render_template
import json
import plotly.utils
from sklearn.neural_network import MLPRegressor
from werkzeug.utils import secure_filename
import pandas as pd
import os
import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pyodbc
from datetime import datetime
import plotly.graph_objects as go
import mpld3

with open ("LagLength.json" ,'r') as e:
    lagvar = e.read()
LagLength = json.loads(lagvar)


app = Flask(__name__)
app.secret_key = '23234sfseadef'
model = pickle.load(open('tseriesmodel.pkl','rb'))
connection_string = 'DRIVER={SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=DBMS;Trusted_Connection=yes;'

@app.route('/totrain',methods=['GET','POST'])
def to_train():
    server_name = r'localhost\\SQLEXPRESS'
    database_name = 'DBMS'
    fitting_values = []
    connection_string = 'DRIVER={ODBC Driver 18 for SQL Server};SERVER=localhost\\SQLEXPRESS;DATABASE=DBMS;Trusted_Connection=yes;TrustServerCertificate=yes'
    connection = pyodbc.connect(connection_string)
    cursor = connection.cursor()
    cursor.execute('select TOP 5000 Timestamp from dbo.maa')
    data = [row[0] for row in cursor.fetchall()]
    if request.method=='POST':
        start = request.form.get("selected_option")
        end = request.form.get("selected_option1")
        print("start:",start)
        print("end:",end)
        cursor.execute(f'select "value" from dbo.maa where Timestamp between ? and ?',start,end)
        train_data = [row[0] for row in cursor.fetchall()]
        print("train_data ",train_data)
        train_data_final = [float(ele) for ele in train_data]
        cursor.close()
        connection.close()
        print("train_data_final:",train_data_final)
        '''All the code to fit the model is as follows:'''
        forGraph = {}
        print("Training the model begins.")
        Timeseries_Data = pd.DataFrame(train_data_final)
        LagLength = 10
        h = 1
        lt = Timeseries_Data.shape[0]
        print('Number of rows in table: ',lt)
        lenTrain = int(round(lt * 0.7))
        lenValidation = int(round(lt * 0.15))
        lenTest = int(lt - lenTrain - lenValidation)
        print('NORMALIZE THE DATA\n')
        normalizedData = minmaxNorm(Timeseries_Data, lenTrain + lenValidation);
        print('Normalized data: ', normalizedData)
        print('Transform the Time Series into Patterns Using Sliding Window')
        print('Fitting the inputs')
        print('lenValidation: ',lenValidation)
        print('lenTest: ',lenTest)
        print('lenTrain',lenTrain)

        X, y = get_Patterns(normalizedData, LagLength, h)
        print('Done with the above')
        model1 = MLPRegressor(hidden_layer_sizes=(100))
        name = 'MLP'
        file1 = './' + str(name) + "_Accuracy.xlsx"
        file2 = './' + str(name) + "_Forecasts.xlsx"
        Forecasts = pd.DataFrame()
        Accuracy = pd.DataFrame()

        ynorm1 = Find_Fitness(X, y, lenValidation, lenTest, model1)
        ynorm = pd.DataFrame(normalizedData.iloc[0:(LagLength + h - 1), 0])
        # ynorm=ynorm.append(ynorm1,ignore_index = True)
        ynorm = pd.concat([ynorm, ynorm1])
        yhat = minmaxDeNorm(Timeseries_Data, ynorm, lenTrain + lenValidation)
        Accuracy.loc[1, 0], Accuracy.loc[1, 1], forGraph = findRMSE(Timeseries_Data, yhat, lenTrain + lenValidation)
        Accuracy.loc[1, 2], Accuracy.loc[1, 3] = findMAE(Timeseries_Data, yhat, lenTrain + lenValidation)
        toGraph = pd.DataFrame(forGraph)
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_name = f'model_{current_datetime}.pkl'
        toGraph.to_csv(f'{model_name}params.csv',index=False,encoding='UTF-8')
        print(Accuracy)
        session['accuracy_values'] = Accuracy.to_dict()

        #Saving the model name in a JSON file so that it may be used to provide options to the user to make predictions.
        with open('model_namesupt.json','r') as file:
            data = json.load(file)
            new_elements=[{model_name:None}]
            data.extend(new_elements)
        with open('model_namesupt.json','w') as file:
            json.dump(data,file,indent=4)

        print('Dumping the model into a pickle file:')
        pickle.dump(model1,open(model_name,'wb'))
        return render_template('totrain.html',data=data)
    cursor.close()
    connection.close()
    return render_template('totrain.html', data=data)
@app.route('/',methods=['GET','POST'])
def home():
    with open('model_namesupt.json','r') as e:
        models_store = e.read()
        models = json.loads(models_store)
        if request.method == 'POST':
            desiredModel = request.form.get("themodel")
            print("Models:",models)
            print("Desired model",desiredModel)
            session['desiredModel'] = desiredModel
            return render_template('home.html',text_boxes=LagLength,models = models,desiredModel=desiredModel)
        return render_template('home.html', text_boxes=LagLength, models=models)
@app.route('/trainingresult',methods=['GET','POST'])
def predict_results():
    Accuracy = session.get('accuracy_values',{})
    print(Accuracy)
    return render_template('predictresult.html',Accuracy=Accuracy)
@app.route('/predictions',methods=['GET','POST'])
def make_predict():
    plot_html = generate_conf()
    modelt = session.get('desiredModel',{})
    print(modelt)
    model = pickle.load(open(modelt, 'rb'))

    if request.method == 'POST':
        if 'csvFile' in request.files:
            print('csv file detected.')
            csv_file = request.files['csvFile']
            if csv_file.filename !='':
                filename = os.path.join(os.getcwd(),secure_filename(csv_file.filename))
                csv_file.save(filename)
                df = pd.read_csv(filename)
                print(df)
                csv_values=df.values.flatten().tolist()
                prediction_result = model.predict([csv_values])
                # os.remove(filename)
                return render_template('result.html', prediction_result=prediction_result,text_boxes=LagLength,plot_html=plot_html)
    parameters = [int(request.form[f'Day{i}']) for i in range(1, LagLength + 1)]
    prediction_result = model.predict([parameters])
    return render_template('result.html', prediction_result=prediction_result,text_boxes=LagLength,plot_html =plot_html)
@app.route('/modelpage',methods=['POST'])
def modelpage():
    return render_template('modelpage.html',text_boxes=LagLength)
@app.route('/uploadpage',methods=['POST'])
def uploadpage():
    return render_template('uploadpage.html',text_boxes=LagLength)
'''The following functions and code is to plot the confidence intervals on the webpage, and to fit the model'''
# originalData should be a Column Vectored DataFrame
def minmaxNorm(originalData, lenTrainValidation):
    # Maximum Value
    max2norm=max(originalData.iloc[0:lenTrainValidation,0])
    # Minimum Value
    min2norm=min(originalData.iloc[0:lenTrainValidation,0])
    lenOriginal=len(originalData)
    normalizedData=np.zeros(lenOriginal)
    normalizedData = []
    #Normalize using Min-Max Normalization
    for i in range (lenOriginal):
        normalizedData.append((originalData.iloc[i]-min2norm)/(max2norm-min2norm))
    return pd.DataFrame(normalizedData)
# split a univariate time series into patterns
def get_Patterns(TSeries, n_inputs,h):
    print('Getting patterns')
    X,z = pd.DataFrame(np.zeros((len(TSeries)-n_inputs-h+1,n_inputs))), pd.DataFrame()
    y_list = []
    print(len(TSeries))
    for i in range(len(TSeries)):
        # find the end of this pattern
        end_ix = i + n_inputs + h - 1

        if end_ix > len(TSeries)-1:
            break
        # gather input and output parts of the pattern
        for j in range(n_inputs):
            X.loc[i,j]=TSeries.iloc[i+j,0]
        i=i+n_inputs
        # y=pd.concat([TSeries.iloc[end_ix],y])
        y_list.append(TSeries.iloc[end_ix])
        print('y_list: ',y_list)
        y = pd.concat(y_list, axis=1).T.reset_index(drop=True)
    print('Reached the end of get_Patterns successfully')
    return X,y
# originalData and forecastedData should be Column Vectored DataFrames
def minmaxDeNorm( originalData, forecastedData, lenTrainValidation):
    # Maximum Value
    max2norm=max(originalData.iloc[0:lenTrainValidation,0])
    # Minimum Value
    min2norm=min(originalData.iloc[0:lenTrainValidation,0])
    lenOriginal=len(originalData)
    denormalizedData=[]
    #De-Normalize using Min-Max Normalization
    for i in range (lenOriginal):
        denormalizedData.append((forecastedData.iloc[i]*(max2norm-min2norm))+min2norm)
    return pd.DataFrame(denormalizedData)



# Timeseries_Data and forecasted_value should be Column Vectored DataFrames
def findRMSE( Timeseries_Data, forecasted_value,lenTrainValidation):
    forGraph = {
        'Actual': [],
        'Predicted': [],
        'Error': []
    }
    l=Timeseries_Data.shape[0]
    lenTest=l-lenTrainValidation
    # RMSE on Train & Validation Set
    trainRMSE=0;
    for i in range (lenTrainValidation):
        forGraph['Actual'].append(Timeseries_Data.iloc[i,0])
        forGraph['Predicted'].append(forecasted_value.iloc[i,0])
        forGraph['Error'].append(forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0])
        trainRMSE=trainRMSE+np.power((forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0]),2)
    trainRMSE=np.sqrt(trainRMSE/lenTrainValidation)
    # RMSE on Test Set
    testRMSE=0;
    for i in range (lenTrainValidation,l,1):
        testRMSE=testRMSE+np.power((forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0]),2)
    testRMSE=np.sqrt(testRMSE/lenTest)
    return trainRMSE, testRMSE, forGraph

# Timeseries_Data and forecasted_value should be Column Vectored DataFrames
def findMAE(Timeseries_Data, forecasted_value,lenTrainValidation):
    l=Timeseries_Data.shape[0]
    lenTest=l-lenTrainValidation
    # MAE on Train & Validation Set
    trainMAE=0;
    for i in range (lenTrainValidation):
        trainMAE=trainMAE+np.abs(forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0])
    trainMAE=(trainMAE/(lenTrainValidation));
    # MAE on Test Set
    testMAE=0;
    for i in range (lenTrainValidation,l,1):
        testMAE=testMAE+np.abs(forecasted_value.iloc[i,0]-Timeseries_Data.iloc[i,0])
    testMAE=(testMAE/lenTest);
    return trainMAE, testMAE

def Find_Fitness(x,y,lenValid,lenTest,model):
    print('entered the fitting function')
    NOP=y.shape[0]
    lenTrain=NOP-lenValid-lenTest
    xTrain=x.iloc[0:lenTrain,:]
    xValid=x.iloc[lenTrain:(lenTrain+lenValid),:]
    xTest=x.iloc[(lenTrain+lenValid):NOP,:]
    yTrain=y.iloc[0:lenTrain,0]
    yValid=y.iloc[lenTrain:(lenTrain+lenValid),0]
    yTest=y.iloc[(lenTrain+lenValid):NOP,0]
    model.fit(xTrain, yTrain)
    print('Model has been fitted')
    yhatNorm=model.predict(x).flatten().reshape(x.shape[0],1)
    print('Returning the dataframe')
    return pd.DataFrame(yhatNorm)


def find_confidence_interval_gaussian(confidenceLevel,sampleMean,sampleStandardError,error,ForecastedData):
    degrees_freedom = len(error)-1  #degree of freedom = sample size-1
    #sampleMean = np.mean(error)    #sample mean
    #sampleStandardError = st.sem(error)  #sample standard error
    confidenceInterval95 = st.norm.interval(confidence=confidenceLevel, loc=sampleMean, scale=sampleStandardError)
    lower95=pd.DataFrame(ForecastedData+confidenceInterval95[0])
    upper95=pd.DataFrame(ForecastedData+confidenceInterval95[1])
    df = pd.concat([lower95, upper95], axis = 1)
    return df


def find_Prediction_Interval_Evaluation(pi, Original, Forecasted, ci):
    # Computation of Prediction Interval Coverage Probability (PICP)
    count = 0
    cc = pd.DataFrame()
    for i in range(0, Original.shape[0], 1):
        dat = Original.iloc[i, 0]
        if dat >= pi.iloc[i, 0] and dat <= pi.iloc[i, 1]:
            count += 1
    PICP = (1 / Original.shape[0]) * count * 100

    # Computation of Prediction Interval Normalized Average Width (PINAW)
    sum = 0
    for i in range(0, pi.shape[0], 1):
        sum += pi.iloc[i, 1] - pi.iloc[i, 0]
    A = np.max(Original) - np.min(Original)
    PINAW = (1 / (A * pi.shape[0])) * sum

    # Computation of Accumulated Width Deviation (AWD)
    sum = 0
    for i in range(0, Original.shape[0], 1):
        dat = Original.iloc[i, 0]
        if dat >= pi.iloc[i, 0] and dat <= pi.iloc[i, 1]:
            sum += 0
        elif dat < pi.iloc[i, 0]:
            sum += (pi.iloc[i, 0] - dat) / (pi.iloc[i, 1] - pi.iloc[i, 0])
        else:
            sum += (dat - pi.iloc[i, 1]) / (pi.iloc[i, 1] - pi.iloc[i, 0])
    AWD = (1 / Original.shape[0]) * sum

    # Computation of Average Coverage Error (ACE)=PICP-PINC
    ACE = PICP - ci
    return PICP, PINAW, AWD, ACE
def generate_conf():
    print('ENTERED GRAPH FN')
    desiredModel = session.get('desiredModel', {})
    print("The desired model is:", desiredModel)
    
    file_path = f"{desiredModel}params.csv"
    print(f"Attempting to read CSV file from: {file_path}")
    try:
        my_data = pd.read_csv(file_path)
        print("CSV data read successfully.")
    except FileNotFoundError:
        print(f"File {file_path} not found.")
        return None
    
    print(f"CSV file contains {len(my_data)} rows.")
    
    # Ensure we have enough data for the given slice
    if len(my_data) < 424:
        print("Insufficient data in the CSV. Adjusting slicing range.")
        # Adjust the slicing range based on the actual number of rows available
        start_idx = max(0, len(my_data) - 114)
        end_idx = len(my_data)
    else:
        start_idx = 310
        end_idx = 424

    print(f"Slicing range: {start_idx} to {end_idx}")
    
    error = pd.DataFrame(my_data.iloc[start_idx:end_idx, 2])
    ForecastedData = pd.DataFrame(my_data.iloc[start_idx:end_idx, 1])
    Original = pd.DataFrame(my_data.iloc[start_idx:end_idx, 0])
    
    print("Data for error, ForecastedData, and Original extracted successfully.")
    print("Error data sample:\n", error.head())
    print("ForecastedData sample:\n", ForecastedData.head())
    print("Original data sample:\n", Original.head())
    
    location = 0
    scale = 5
    conf95 = find_confidence_interval_gaussian(0.95, location, scale, error, ForecastedData)
    print("95% confidence interval calculated.")
    conf90 = find_confidence_interval_gaussian(0.90, location, scale, error, ForecastedData)
    print("90% confidence interval calculated.")
    conf80 = find_confidence_interval_gaussian(0.80, location, scale, error, ForecastedData)
    print("80% confidence interval calculated.")
    conf70 = find_confidence_interval_gaussian(0.70, location, scale, error, ForecastedData)
    print("70% confidence interval calculated.")
    conf60 = find_confidence_interval_gaussian(0.60, location, scale, error, ForecastedData)
    print("60% confidence interval calculated.")
    
    # Create a Plotly figure
    fig = go.Figure()
    
    # Add traces for confidence intervals
    fig.add_trace(go.Scatter(
        x=conf90.index,
        y=conf90.iloc[:, 0],
        fill='toself',
        fillcolor='rgba(207, 0, 0, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='90% CI'
    ))
    
    fig.add_trace(go.Scatter(
        x=conf80.index,
        y=conf80.iloc[:, 0],
        fill='toself',
        fillcolor='rgba(175, 0, 0, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='80% CI'
    ))
    
    fig.add_trace(go.Scatter(
        x=conf70.index,
        y=conf70.iloc[:, 0],
        fill='toself',
        fillcolor='rgba(143, 0, 0, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='70% CI'
    ))
    
    fig.add_trace(go.Scatter(
        x=conf60.index,
        y=conf60.iloc[:, 0],
        fill='toself',
        fillcolor='rgba(111, 0, 0, 0.3)',
        line=dict(color='rgba(255,255,255,0)'),
        name='60% CI'
    ))
    
    print("Confidence intervals added to the Plotly figure.")
    
    # Add traces for Forecasts and True Values
    fig.add_trace(go.Scatter(
        x=ForecastedData.index,
        y=ForecastedData.iloc[:, 0],
        mode='lines',
        name='Forecasts',
        line=dict(color='rgba(0, 0, 0, 0.7)')
    ))
    
    fig.add_trace(go.Scatter(
        x=Original.index,
        y=Original.iloc[:, 0],
        mode='lines',
        name='True Values',
        line=dict(color='rgba(0, 0, 0, 0.7)', dash='dash')
    ))
    
    print("Forecasted and True Values added to the Plotly figure.")
    
    # Update layout
    fig.update_layout(
        height=500,
        width=1000,
        showlegend=True,
        legend=dict(x=0, y=1),
        xaxis=dict(title='Time / Month'),
        yaxis=dict(title='Monthly Crude Oil Price'),
    )
    
    figjson = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    print("Figure generated successfully.")
    return figjson


if __name__ == "__main__":
    app.run(debug=True)

