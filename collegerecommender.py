# Artificial Neural Network

import matplotlib.pyplot as plt
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

class College_Recommender():
    def __init__(self, dataset_file, month,field,sat,gre,awa,toefl,ielts,work,paper,loan,international,grade):
        self.month = month
        self.field = field
        self.sat = sat
        self.gre = gre
        self.awa = awa
        self.toefl = toefl
        self.ielts = ielts
        self.work = work
        self.paper = paper
        self.loan = loan
        self.international = international
        self.grade = grade
        self.dataset_file = dataset_file

        self.fetch_data()
        self.encode_data()
        self.fix_missing_data()
        self.split_data()
        self.classifier = self.train_model()
        self.array = self.configure_input()
        self.predict_acceptance(self.classifier, self.array)

    def fetch_data(self):
        dataset = pd.read_csv(self.dataset_file, error_bad_lines=False, encoding = "ISO-8859-1")
        self.X = dataset.iloc[:, 1:13].values
        self.y = dataset.iloc[:, 13].values
        self.X[:,0] = np.random.randint(1, 12, size=len(self.X[:,0]))
    
    def encode_data(self):
        labelencoder_X_1 = LabelEncoder()
        self.X[:, 1] = labelencoder_X_1.fit_transform(self.X[:, 1])
        self.classes = dict(zip(labelencoder_X_1.classes_, labelencoder_X_1.transform(labelencoder_X_1.classes_)))

        labelencoder_y_1 = LabelEncoder()
        self.y = labelencoder_y_1.fit_transform(self.y)
        self.classes2 = dict(zip(labelencoder_y_1.classes_, labelencoder_y_1.transform(labelencoder_y_1.classes_)))

    def fix_missing_data(self):
        imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
        imputer = imputer.fit(self.X[:, :])
        self.X[:, :] = imputer.transform(self.X[:, :])
        self.X = self.X.astype(float)
        self.y = self.y.astype(int) 
        self.y = self.y.reshape(-1, 1)


    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.5)
        self.sc = StandardScaler()
        self.X_train = self.sc.fit_transform(self.X_train)
        self.X_test = self.sc.transform(self.X_test)
        self.y_train = self.sc.fit_transform(self.y_train)
        self.y_test = self.sc.transform(self.y_test)

    def train_model(self):
        keras.backend.clear_session()
        
        classifier = Sequential()
    	# Adding the input layer and the first hidden layer
        classifier.add(Dense(units = 13, kernel_initializer = 'glorot_uniform', activation = 'relu'))

        # Dense Layer
        classifier.add(Dense(1, kernel_initializer='normal'))
        
        # Compiling the ANN
        classifier.compile(loss = 'mean_squared_error', optimizer='adam')
        
        # Fitting the ANN to the Training set
        classifier.fit(self.X_train, self.y_train, batch_size = 10, epochs = 100)

        # Predicting the Test set results
        self.y_pred = classifier.predict(self.X_test)
        self.y_pred = (self.y_pred > 0.5)
        return classifier
    
    def configure_input(self):
        array = np.array([[self.month,self.field,self.sat,self.gre,self.awa,self.toefl,self.ielts,self.work,self.paper,self.loan,self.international,self.grade]])

        month = {
            "Jan" : 1,
            "Feb" : 2,
            "Mar" : 3,
            "Apr" : 4,
            "May" : 5,
            "Jun" : 6,
            "Jul" : 7,
            "Aug" : 8,
            "Sep" : 9,
            "Oct" : 10,
            "Nov" : 11,
            "Dec" : 12
        }

        inter = {
            "yes" : 1,
            "no" : 0
            }

        word  = str(array[:,0])
        word = word.replace('[', '')
        word = word.replace(']', '')
        word = word.replace("'", "")
        ans = month[word]
        array[:,0] = ans

        word  = str(array[:,10])
        word = word.replace('[', '')
        word = word.replace(']', '')
        word = word.replace("'", "")
        ans = inter[word]
        array[:,10] = ans
        
        word  = str(array[:,1])
        word = word.replace('[', '')
        word = word.replace(']', '')
        word = word.replace("'", "")
        ans = self.classes[word]
        array[:,1] = ans
        
        OldValue = float(array[:,2])
        NewMin = 130
        OldMin = 400
        OldMax = 1600
        NewMax = 340
        NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        array[:,2] = NewValue
        
        return array

    def predict_acceptance(self, classifier, array):
        new_prediction = classifier.predict(self.sc.transform(array))
        new_prediction = int((abs(self.sc.inverse_transform(new_prediction))))


        finale = [key for key, val in self.classes2.items() if val == new_prediction]
        sentence = []

        word  = str(finale)
        word = word.replace('[', '')
        word = word.replace(']', '')
        response = word.replace("'", "")
        sentence.append(response)
        final = ""    
        y = 1;

        for element in sentence:
            final = element + "  " + final
            y+=1
        new = []
        for key in self.classes2.keys():
            new.append(key)
        self.final = final
    
    def return_respose(self):
        return self.final


