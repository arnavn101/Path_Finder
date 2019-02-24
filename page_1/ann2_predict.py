
def ann2_predict(mon,field,sat,gre,awa,toefl,ielts,work,paper,loan,international,grade):

    import tensorflow
    import numpy
    import pandas
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.wrappers.scikit_learn import KerasRegressor
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import cross_val_predict

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    # load dataset
    # split into input (X) and output (Y) variables

    # load dataset
    # Importing the libraries

    # Importing the dataset
    dataset = pd.read_csv('data_set.csv',error_bad_lines=False, encoding = "ISO-8859-1")


    X = dataset.iloc[:, 1:13].values
    y = dataset.iloc[:, 13].values

    X[:,0] = np.random.randint(1, 12, size=len(X[:,0]))


    # Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    classes = dict(zip(labelencoder_X_1.classes_, labelencoder_X_1.transform(labelencoder_X_1.classes_)))




    labelencoder_y_1 = LabelEncoder()
    y = labelencoder_y_1.fit_transform(y)
    classes2 = dict(zip(labelencoder_y_1.classes_, labelencoder_y_1.transform(labelencoder_y_1.classes_)))


    y = y.astype(int)



    #Missing Data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, :])
    X[:, :] = imputer.transform(X[:, :])
    X = X.astype(float)

    y = y.reshape(-1, 1)
    # Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


    y_train = sc.fit_transform(y_train)
    y_test = sc.transform(y_test)


    
    sentence = []
    x = 0;
    number = 1
    choice = number

    	# define base model
   
    	# create model
    classifier = Sequential()
    classifier.add(Dense(units = 13, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dense(1, kernel_initializer='normal'))

        	# Compile model
    classifier.compile(loss='mean_squared_error', optimizer='adam')
    
    classifier.fit(X_train,y_train,batch_size = 10, epochs = 100)
            
    y_pred = classifier.predict(X_test)
    from sklearn.model_selection import GridSearchCV
    
    
    
    array = np.array([[mon,field,sat,gre,awa,toefl,ielts,work,paper,loan,international,grade]])
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
    ans = classes[word]
    array[:,1] = ans
    
    
    OldValue = float(array[:,2])
    NewMin = 130
    OldMin =400
    OldMax =1600
    NewMax = 340
    NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    array[:,2] = NewValue
    
    
    new_prediction = classifier.predict(sc.transform(array))
    new_prediction = int((abs(sc.inverse_transform(new_prediction))))
    
    
    
    
    finale = [key for key, val in classes2.items() if val == new_prediction]
    word  = str(finale)
    word = word.replace('[', '')
    word = word.replace(']', '')
    response = word.replace("'", "")
    sentence.append(response)
    x = x + 1
    final = ""    
    y = 1;
    for element in sentence:
        final = element + "  " + final
        y+=1
    import random
    new = []
    for key in classes2.keys():
      new.append(key)
    
    return(" You are most likely to get accepted into " + final)
#ann_predict(3, 'Jan', 'CS', 1400, 111, 10, 19, 101,1,1,1,'yes',90)