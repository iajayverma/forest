from flask import Flask,request,jsonify,render_template
import pickle 
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

app=Flask(__name__)

ridge=pickle.load(open('models/ridge.pkl','rb'))
standard_scaler=pickle.load(open('models/sacler.pkl','rb'))


### route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/algerian',methods=['GET','POST'])
def predictdata_point():
    if request.method=='POST':
        temperature = float(request.form['Temperature'])
        rh = float(request.form['RH'])
        ws = float(request.form['WS'])
        rain = float(request.form['Rain'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])
        classes = float(request.form['Classes'])
        region = float(request.form['Region'])
        new_data_scaled=standard_scaler.transform([[temperature,rh,ws,rain,ffmc,dmc,isi,classes,region]])
        result=ridge.predict(new_data_scaled)

        return render_template('home.html',result=result[0])


    else:
       return render_template('home.html')
    


if __name__ =="__main__":
    app.run()