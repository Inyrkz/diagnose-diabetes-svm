from flask import Flask, request
from flask_restful import Api, Resource
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler


# instantiate Flask Rest Api
app = Flask(__name__)
api = Api(app)

# load the pickled model and X_train
X_train = pickle.load(open('X_train.sav', 'rb'))
model = pickle.load(open('model.sav', 'rb'))

# feature scale data after fitting scalar object to pickled training set
sc = StandardScaler()
scaled_X_train = sc.fit_transform(X_train)


# Create class for Api Resource
class Records(Resource):
    def get(self):
        # get request that returns the JSON format for API request
        return {"JSON data format": {"Pregnancies": 5, 
                                     "Glucose": 32,
                                     "BloodPressure": 43,
                                     "SkinThickness": 23,
                                     "Insulin":22,
                                     "BMI": 33.6,
                                     "DiabetesPedigreeFunction":  1.332,
                                     "Age": 50 
                                    }
                }, 200

    def post(self):
        # post request
        # make model and scalar object global variables
        global model
        global sc
        # it gets patient's record and returns the ML model's prediction
        data = request.get_json()
        
        try:
            pregnancies = int(data["Pregnancies"])
            glucose = int(data["Glucose"])
            bp = int(data["BloodPressure"])
            st = int(data["SkinThickness"])
            insulin = int(data["Insulin"])
            bmi = float(data["BMI"])
            dpf = float(data["DiabetesPedigreeFunction"])
            age = int(data["Age"])
            
            # model expects a 2D array
            new_record = np.array([[pregnancies, glucose, bp, st, insulin, bmi, dpf, age]])
            
            # feature scale the data
            scaled_data = sc.transform(new_record)
            
            # pass scaled data to model for prediction
            new_pred = model.predict(scaled_data)[0]
            
            # dictionary containing the diagnosis with the key as the model's prediction
            diagnosis = {0: 'Your Result is Normal', 
                         1: 'Diabetes Detected'
                        }
            # get corresponding value from the diagnosis dictionary (using the model prediction as the key)
            result = diagnosis.get(new_pred)
            return {'Diagnosis': result}, 200
        except:
            # if client sends the wrong request or data type then return correct format
            return {'Wrong data sent to server! Please use this JSON format': {"Pregnancies": 5, 
                                              "Glucose": 32,
                                              "BloodPressure": 43,
                                              "SkinThickness": 23,
                                              "Insulin":22,
                                              "BMI": 33.6,
                                              "DiabetesPedigreeFunction":  1.332,
                                              "Age": 50 
                                             }}, 500

api.add_resource(Records, '/')
# app.run(port=5000, debug=True)
