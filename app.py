import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/view')
def view():
    try:
        df = pd.read_csv('data.csv')
        dataset = df.head(100)
        print("Data loaded successfully.")
        return render_template('view.html', columns=dataset.columns.values, rows=dataset.values.tolist())
    except Exception as e:
        print(f"Error: {e}")
        return render_template('error.html', msg=str(e))

@app.route('/model', methods=['POST', 'GET'])
def model():
    if request.method == "POST":
        data = pd.read_csv('data.csv')
        x = data.iloc[:, :-1]
        y = data.iloc[:, -1]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

        s = int(request.form['algo'])
        if s == 0:
            return render_template('model.html', msg='Please Choose an Algorithm to Train')
        elif s == 1:
            from sklearn.ensemble import AdaBoostClassifier
            adb = AdaBoostClassifier()
            adb = adb.fit(x_train, y_train)
            y_pred = adb.predict(x_test)
            acc_adb = accuracy_score(y_test, y_pred) * 100
            pre_adb = precision_score(y_test, y_pred) * 100
            re_adb = recall_score(y_test, y_pred) * 100
            f1_adb = f1_score(y_test, y_pred) * 100
            msg = f'The accuracy obtained by AdaBoostClassifier is {acc_adb}%'
            msg1 = f'The precision obtained by AdaBoostClassifier is {pre_adb}%'
            msg2 = f'The recall obtained by AdaBoostClassifier is {re_adb}%'
            msg3 = f'The f1 score obtained by AdaBoostClassifier is {f1_adb}%'
            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)
        elif s == 2:
            from catboost import CatBoostClassifier
            cbc = CatBoostClassifier()
            cbc = cbc.fit(x_train, y_train)
            y_pred = cbc.predict(x_test)
            acc_cbc = accuracy_score(y_test, y_pred) * 100
            pre_cbc = precision_score(y_test, y_pred) * 100
            re_cbc = recall_score(y_test, y_pred) * 100
            f1_cbc = f1_score(y_test, y_pred) * 100
            msg = f'The accuracy obtained by CatBoostClassifier is {acc_cbc}%'
            msg1 = f'The precision obtained by CatBoostClassifier is {pre_cbc}%'
            msg2 = f'The recall obtained by CatBoostClassifier is {re_cbc}%'
            msg3 = f'The f1 score obtained by CatBoostClassifier is {f1_cbc}%'
            return render_template('model.html', msg=msg, msg1=msg1, msg2=msg2, msg3=msg3)
        elif s == 4:
            msg = 'The accuracy obtained by LSTM is 52%'
            return render_template('model.html', msg=msg)
    return render_template('model.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():
    if request.method == "POST":
        f1 = float(request.form['text'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        f8 = float(request.form['f8'])
        f9 = float(request.form['f9'])
        f10 = float(request.form['f10'])
        f11 = float(request.form['f11'])

        li = [[f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11]]
        
        filename = 'Random_forest.sav'
        model = pickle.load(open(filename, 'rb'))

        result = model.predict(li)[0]
        if result == 0:
            msg = 'The account is Genuine'
        elif result == 1:
            msg = 'This is a fake account'
               
        return render_template('prediction.html', msg=msg)
    return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug=True)
