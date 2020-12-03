import joblib
import numpy as np

from flask import Flask
from flask import jsonify
from flask import render_template
from flask import request, redirect, url_for

app = Flask(__name__)


@app.route('/calc',methods=['GET', 'POST'])
def formulario():
    if request.method == 'POST':
        ndvi = request.form['ndvi']
        twi = request.form['twi']
        slope = request.form['slope']
        dem = request.form['dem']
        
        X_test = np.array([ndvi,twi,slope,dem])
        prediction = model.predict(X_test.reshape(1,-1))
        
        next = request.args.get('next', None)
        if next:
            return redirect(next)
        return jsonify({'prediccion' : list(prediction)})    
        return redirect(url_for('index')) 
        
       
    return render_template('form.html')
    return jsonify({'prediccion' : list(prediction)})

   

if __name__ == "__main__":
    model = joblib.load('./models/rand_est.pkl')
    app.run(port=7070)