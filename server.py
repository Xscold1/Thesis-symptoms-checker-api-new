
from lib2to3.pytree import Base
import pandas as pd
import numpy as np
from fastapi import FastAPI, Path, Request
from typing import Any, Optional, List, Dict, AnyStr
from pandas import array
from pydantic import BaseModel
from joblib import dump, load
from scipy.stats import mode
import operator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy import mean, stats
import json
import matplotlib.pyplot as plt
import seaborn as sns
import math

PROBABILITY_THRESHOLD = 0.1
TEST_DATA_PERCENTAGE = 20
# The more the dataset the more the model will be accurate
NUMBER_OF_DATASET_PER_DISEASE = 10
app = FastAPI()

class InputValue(BaseModel):
    input: List[str]
    additionalInfo: Optional[Dict[AnyStr, Any]] = None
    age: Optional[int] = None
    gender: Optional[int] = None
    
class DatasetFormat(Base):
    input: Dict
    
@app.post("/predict")
def predict(body: InputValue):
    try:
        svm_model = load('./models/svm_model')
        random_forest_model = load('./models/random_forest_model')
        logistic_regression_model = load('./models/logistic_regression_model')
        data_dict = load('./models/data_dict')
        
        input = body.input

        if len(input) <= 0:
            return 'Please provide symptoms'
    
        # creating input data for the models
        input_data = [0] * len(data_dict["symptom_index"])

        total_supported_symptoms = 0
        
        def textToTitle(n):
            return n.title()
        
        input = list(map(textToTitle,input))

        for symptom in input:
            symptom = symptom.title() 
            if symptom in data_dict["symptom_index"]:
                index = data_dict["symptom_index"][symptom.title()] # IMPORTANT make all first letter in all words capitalize
                input_data[index] = 1
                total_supported_symptoms += 1

        if total_supported_symptoms <= 0:
            return 'Looking good! Your symptoms does not match any data on our system'
        
        if body.additionalInfo:
        # Inputting for additional info
            for info in body.additionalInfo:
                if body.additionalInfo[info] == True:
                    decoded_info = info.decode('utf-8')
                    formatted = decoded_info.lower().replace(" ", "_").replace(",", "")
                    
                    index = data_dict["symptom_index"][formatted]
                    # # Adding 1  mmeans the symptom is yes
                    input_data[index] = 1
        
        if body.age:
            age_index = data_dict["symptom_index"]["Age"]
            print('age_index: ', age_index)
            input_data[age_index] = body.age
        
        if body.gender:
            gender_index = data_dict["symptom_index"]['Gender']
            input_data[gender_index] = body.gender

        # reshaping the input data and converting it
        # into suitable format for model predictions in scikit learn
        input_data = np.array(input_data).reshape(1,-1)
        
        # generating individual outputs
        random_forest_probability = random_forest_model.predict_proba(input_data)[0]
        svm_prediction_probability = svm_model.predict_proba(input_data)[0]
        logistic_regression_probability = logistic_regression_model.predict_proba(input_data)[0]

        combined_models_probability =  mean([random_forest_probability,logistic_regression_probability, svm_prediction_probability], axis = 0)
        predicted_diseases_dict = {}
        
        for i, x in enumerate(data_dict["predictions_classes"]):
            #  Creating dictionary that have key of disease and it's percentage
            # We will only push items that are greater than the threshold
            if combined_models_probability[i] > PROBABILITY_THRESHOLD:
                predicted_diseases_dict[x] = combined_models_probability[i]
        # disease to symptom json
        if len(predicted_diseases_dict) <= 0:
            return 'Looking good! Your symptoms does not match any data on our system'
        
        # Sorting the result to decensing order
        sorted_diseases_dict = dict(sorted(predicted_diseases_dict.items(), key=operator.itemgetter(1),reverse=True))
        with open('./models/disease_symptoms_list.json', 'r') as f:
            data = json.load(f)
            
        keysToRemove = []
        
        # Check if the predicted disease and the provided symptoms is existing in the disease to symptoms map, this is a FILTER
        for key in sorted_diseases_dict:
            isExisting = any(map(lambda each: each in input, data[key]))
            if not isExisting:
                keysToRemove.append(key)

        for key in keysToRemove:
            del sorted_diseases_dict[key]

        return {
                'status': 'OK',
                'status_code': 200,
                'message': 'Successfully retrived predictions',
                'response': sorted_diseases_dict
        }
    except Exception as e:
        print('error: ', e)
        return {
            'status': 'Interval Server Error',
            'status_code': 500,
            'message': 'Something went wrong!'
        }

@app.post("/create-dataset")
async def create_dataset(request: Dict[AnyStr, Any]):
    try:
        # Loads the dataset format from thesis-api
        disease_symptom = json.loads(request[b'input'])
        additional_info = request[b'additionalInfo']
        dict = {}
        dict['age'] = []
        dict['gender'] = []
        # Conditional to use mock dataset or real dataset from the database
        for index, key in enumerate(disease_symptom):
            for symptom in disease_symptom[key]:
                formatted = symptom.lower().replace(" ", "_").replace(",", "")
                dict[formatted] = []
        # # additional info
        for info in additional_info:
            info = info.lower().replace(" ", "_").replace(",", "")
            dict[info] = []

        dict['prognosis'] = []
        info = pd.DataFrame(dict)  
        
        #  Create Dataset
        info.to_csv('./dataset/symptom_checker_dataset.csv', index=False)  

        # Insert Initial Dataset
        symptoms = info.columns.values
        symptom_index = {}
        for index, value in enumerate(symptoms):
            symptom_index[value] = index
            
        data_dict = {
            "symptom_index": symptom_index,
        }

        for index, key in enumerate(disease_symptom):
            input_data = [0] * len(data_dict["symptom_index"])
            
            for j, symptom in enumerate(disease_symptom[key]):
                for symptom in disease_symptom[key]:
                
                    formatted = symptom.lower().replace(" ", "_").replace(",", "")
                    index = data_dict["symptom_index"][formatted]
                    # Adding 1  mmeans the symptom is yes
                    input_data[index] = 1
                    
            prognosisIndex = data_dict["symptom_index"]['prognosis']
                
            input_data[prognosisIndex] = key
            for i in range(NUMBER_OF_DATASET_PER_DISEASE):
                df = pd.DataFrame([input_data])
                df.to_csv('./dataset/symptom_checker_dataset.csv', mode='a', index=False, header=False)  
        
        return {
                'status': 'OK',
                'status_code': 200,
                'message': 'Successfully created dataset'
        }
    except Exception as e:
        print('e: ', e)
        return {
            'status': 'Interval Server Error',
            'status_code': 500,
            'message': 'Something went wrong!'
        }
    

@app.post("/add-dataset")
def add_dataset(input: List[str],additionalInfo: Dict[AnyStr, Any], inputDisease: str, age: Optional[int] = None, gender: Optional[int] = None):
    try:
        created = pd.read_csv('dataset/symptom_checker_dataset.csv')
        created.head()
        symptoms = created.columns.values
        symptom_index = {}
        
        for index, value in enumerate(symptoms):
            symptom_index[value] = index
            
        data_dict = {
            "symptom_index": symptom_index,
        }

        input_data = [0] * len(data_dict["symptom_index"])

        # Inputting for symptoms
        for symptom in input:
            formatted = symptom.lower().replace(" ", "_").replace(",", "")
            
            index = data_dict["symptom_index"][formatted]
            # Adding 1  mmeans the symptom is yes
            input_data[index] = 1
        
        # Inputting for additional info
        for info in additionalInfo:
            if additionalInfo[info] == True:
                decoded_info = info.decode('utf-8')
                formatted = decoded_info.lower().replace(" ", "_").replace(",", "")
                
                index = data_dict["symptom_index"][formatted]
                # # Adding 1  mmeans the symptom is yes
                input_data[index] = 1
        
        if age:
            age_index = data_dict["symptom_index"]['age']
            input_data[age_index] = age
        
        if gender:
            gender_index = data_dict["symptom_index"]['gender']
            input_data[gender_index] = gender
        
        prognosisIndex = data_dict["symptom_index"]['prognosis']

        # Get the prognosis index and input the result disease
        input_data[prognosisIndex] = inputDisease
        
        # Make data frame of above data
        df = pd.DataFrame([input_data])
        df.to_csv('dataset/symptom_checker_dataset.csv', mode='a', index=False, header=False)

        return {
            'status': 'OK',
            'status_code': 200,
            'message': 'Successfully added dataset'
        }
    except Exception as e:
        print('e: ', e)
        return {
            'status': 'Interval Server Error',
            'status_code': 500,
            'message': 'Something went wrong!'
        }
    

@app.get("/create-model")
def create_model():
    try:
        DATA_PATH = "./dataset/symptom_checker_dataset.csv"
        data = pd.read_csv(DATA_PATH).dropna(axis = 1)
        encoder = LabelEncoder()
        data["prognosis"] = encoder.fit_transform(data["prognosis"])
        X = data.iloc[:,:-1] # get all up to the end except the last index
        y = data.iloc[:, -1] # get the last index

        final_svm_model = SVC(probability=True)
        final_rf_model = RandomForestClassifier(random_state=18)
        final_lr_model = LogisticRegression()
        final_svm_model.fit(X, y)
        final_rf_model.fit(X, y)
        final_lr_model.fit(X, y)
        dump(final_rf_model, filename="./models/random_forest_model")
        dump(final_svm_model, filename="./models/svm_model")
        dump(final_lr_model, filename="./models/logistic_regression_model")
        symptoms = X.columns.values
        symptom_index = {}
        
        for index, value in enumerate(symptoms):
            symptom = " ".join([i.capitalize() for i in value.split("_")])
            symptom_index[symptom] = index

        data_dict = {
            "symptom_index": symptom_index, # symptom converted indexes
            "predictions_classes": encoder.classes_ # list all the predictions classes
        }

        dump(data_dict, filename="./models/data_dict") # Exporting data_dict for future use

        return {
            'status': 'OK',
            'status_code': 200,
            'message': 'Successfully created disease symptoms list'
        }
    except Exception as e:
        print('e: ', e)
        return {
            'status': 'Interval Server Error',
            'status_code': 500,
            'message': 'Something went wrong!'
        }

@app.post("/create_disease_symptoms")
def create_disease_symptoms(request: Dict[AnyStr, Any]):
    try:
        disease_symptom = json.loads(request[b'input'])
        dump = json.dumps(disease_symptom) 

        with open("./models/disease_symptoms_list.json", "w") as outfile:
            outfile.write(dump)

        return {
            'status': 'OK',
            'status_code': 200,
            'message': 'Successfully created disease symptoms list'
        }
    except Exception as e:
        print('e: ', e)
        return {
            'status': 'Interval Server Error',
            'status_code': 500,
            'message': 'Something went wrong!'
        }

@app.get("/variable-to-disease-relationship")
def variable_to_disease_relationship(variable: str, disease: str):
    try:
        # https://stackoverflow.com/questions/42579908/use-corr-to-get-the-correlation-between-two-columns
        variable_formatted =  variable.lower().replace(" ", "_").replace(",", "")
        disease_formatted = disease.title()
        
        DATA_PATH = "./dataset/symptom_checker_dataset.csv"
        data = pd.read_csv(DATA_PATH).dropna(axis = 1)

        data = data.dropna(axis=1)

        try:
            x = data[variable_formatted] # get the last index
            y = data['prognosis'].apply(lambda x: 1 if x==disease_formatted else 0)
        except:
            return {
            'status': 'Bad Request',
            'status_code': 400,
            'message': 'Variable is not supported please try again another variable'
            }

        try:
            # If all the x values are identical this will return error
            slope, intercept, r, p, std_err = stats.linregress(x, y)
            
        except Exception as e:
            print('e: ', e)
            if isinstance('ValueError', str):
                return {
                'status': 'OK',
                'status_code': 200,
                'message': 'Successfully retrived the variable to disease relationship',
                'response': 1
                }
            else:
                return {
                'status': 'Interval Server Error',
                'status_code': 500,
                'message': 'Something went wrong!'
            }
            
        return {
            'status': 'OK',
            'status_code': 200,
            'message': 'Successfully retrived the variable to disease relationship',
            'response': r
        }
    except Exception as e:
        print('e: ', e)
        return {
            'status': 'Interval Server Error',
            'status_code': 500,
            'message': 'Something went wrong!'
        }
        
@app.get("/models-accuracy")
def models_accuracy():
    try:
        accuracy_dict = {
            "cross_validation_mean_score":{},
            "accuracy_score": {
                "models":{
                    "SVM Classifier": {},
                    "Logistic Regression": {},
                    "Random Forest Classifier": {}
                },
                "dataset": {
                    "train": {},
                    "test": {}
                }
            }
        }
        
        DATA_PATH = "./dataset/symptom_checker_dataset.csv"
        data = pd.read_csv(DATA_PATH).dropna(axis = 1)
        disease_counts = data["prognosis"].value_counts()

        encoder = LabelEncoder()
        data["prognosis"] = encoder.fit_transform(data["prognosis"])
        data.tail()
        
        # ============================ CROSS VALIDATION SCORE ============================
        
        X = data.iloc[:,:-1] # get all up to the end except the last index
        y = data.iloc[:, -1] # get the last index

        # Defining scoring metric for k-fold cross validation
        def cv_scoring(estimator, X, y):
            return accuracy_score(y, estimator.predict(X))

        # Initializing Models
        models = {
            "SVM Classifier": SVC(),
            "Random Forest Classifier":  RandomForestClassifier(random_state=18),
            "Logistic Regression": LogisticRegression()
        }

        # Producing cross validation score for the models
        for model_name in models:
            accuracy_dict['cross_validation_mean_score'][model_name] = {}
            model = models[model_name]
            scores = cross_val_score(model, X, y, cv = 10,
                                    n_jobs = -1,
                                    scoring = cv_scoring)

            accuracy_dict['cross_validation_mean_score'][model_name] = np.mean(scores)

        # ============================ ACCURACY SCORE ============================
        X_train, X_test, y_train, y_test =train_test_split(
        X, y, test_size = 0.2, random_state = 24)
        
        accuracy_dict['accuracy_score']['dataset']['total_number_of_dataset'] = len(X)
        accuracy_dict['accuracy_score']['dataset']['train']['train_data_percentage'] = str(100 - TEST_DATA_PERCENTAGE) + '%'
        accuracy_dict['accuracy_score']['dataset']['test']['test_data_percentage'] = str(TEST_DATA_PERCENTAGE) + '%'
        accuracy_dict['accuracy_score']['dataset']['train']['train_data_total'] = math.ceil(len(X) * (100 - TEST_DATA_PERCENTAGE) / 100)
        accuracy_dict['accuracy_score']['dataset']['test']['test_data_total'] = math.ceil(len(X) * TEST_DATA_PERCENTAGE / 100)
        
        # Training and testing SVM Classifier
        svm_model = SVC()
        svm_model.fit(X_train, y_train)
        preds = svm_model.predict(X_test)
        
        accuracy_dict['accuracy_score']['models']['SVM Classifier']['train_data_accuracy'] = accuracy_score(y_train, svm_model.predict(X_train))*100
        accuracy_dict['accuracy_score']['models']['SVM Classifier']['test_data_accuracy'] = accuracy_score(y_test, preds)*100
        
        # Training and testing Logistic Regression
        lr_model = LogisticRegression()
        lr_model.fit(X_train, y_train)
        preds = lr_model.predict(X_test)
        
        accuracy_dict['accuracy_score']['models']['Logistic Regression']['train_data_accuracy'] = accuracy_score(y_train, lr_model.predict(X_train))*100
        accuracy_dict['accuracy_score']['models']['Logistic Regression']['test_data_accuracy'] = accuracy_score(y_test, preds)*100
        
        # Training and testing Random Forest Classifier
        rf_model = RandomForestClassifier(random_state=18)
        rf_model.fit(X_train, y_train)
        preds = rf_model.predict(X_test)

        accuracy_dict['accuracy_score']['models']['Random Forest Classifier']['train_data_accuracy'] = accuracy_score(y_train, rf_model.predict(X_train))*100
        accuracy_dict['accuracy_score']['models']['Random Forest Classifier']['test_data_accuracy'] = accuracy_score(y_test, preds)*100

        # ============================ COMBINED MODELS ACCURACY SCORE ============================
        
        # Training the models on whole data
        final_svm_model = SVC(probability=True)
        final_rf_model = RandomForestClassifier(random_state=18)
        final_lr_model = LogisticRegression()

        final_svm_model.fit(X_train, y_train)
        final_rf_model.fit(X_train, y_train)
        final_lr_model.fit(X_train, y_train)
        
        # Reading the test data
        test_data = pd.read_csv("./dataset/symptom_checker_dataset.csv").dropna(axis=1)
        
        test_X = test_data.iloc[:, :-1]
        test_Y = encoder.transform(test_data.iloc[:, -1])
        
        # Making prediction by take mode of predictions
        # made by all the classifiers
        svm_preds = final_svm_model.predict(test_X)
        rf_preds = final_rf_model.predict(test_X)
        lr_preds = final_lr_model.predict(test_X)

        final_preds = [mode([i,j,k])[0][0] for i,j,
                    k in zip(svm_preds, rf_preds, lr_preds)]

        accuracy_dict['combined_models_accuracy_score'] = accuracy_score(test_Y, final_preds) * 100
        
        return {
            'status': 'OK',
            'status_code': 200,
            'message': 'Successfully retrived the models accuracy',
            'response': accuracy_dict
        }
    except Exception as e:
        print(e)
        return {
            'status': 'Interval Server Error',
            'status_code': 500,
            'message': 'Something went wrong!'
        }
        
@app.get('/')
def index():
    return {"name":"First Data"}