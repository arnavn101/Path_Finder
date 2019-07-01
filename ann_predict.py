# Artificial Neural Network


# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Scale Data: NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

# array = np.array([['Dec', 'CS', 1600, 111, 10, 19, 101,1,1,1,'yes',90, 'Carnegie Mellon University']])


def ann_predict(month,field,sat,gre,awa,toefl,ielts,work,paper,loan,international,grade,university ):
	# Importing the libraries
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    
    import numpy as np
    	# Importing the dataset
    dataset = pd.read_csv('data_set.csv',error_bad_lines=False, encoding = "ISO-8859-1")
    
    	  
    X = dataset.iloc[:, 1:14].values
    y = dataset.iloc[:, 15].values
    
    X[:,0] = np.random.randint(1, 12, size=len(X[:,0]))
    
    
    	# Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    classes = dict(zip(labelencoder_X_1.classes_, labelencoder_X_1.transform(labelencoder_X_1.classes_)))
    
    
    
    
    labelencoder_X_2 = LabelEncoder()
    X[:, 12] = labelencoder_X_2.fit_transform(X[:, 12])
    classes2 = dict(zip(labelencoder_X_2.classes_, labelencoder_X_2.transform(labelencoder_X_2.classes_)))
    
    
    labelencoder_y_1 = LabelEncoder()
    y = labelencoder_y_1.fit_transform(y)
    y = y.astype(int)
    
    	#Missing Data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, :])
    X[:, :] = imputer.transform(X[:, :])
    X = X.astype(float)
    
    
    	# Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)
    
    	# Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    	#  make the ANN!
    
    	# Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    	# Initialising the ANN
    classifier = Sequential()
    
    	# Adding the input layer and the first hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 13))
    
    	# Adding the second hidden layer
    classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    
    	# Adding the output layer
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    
    	# Compiling the ANN
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    	# Fitting the ANN to the Training set
    classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
    
    	# Part 3 - Making predictions and evaluating the model
    
    	# Predicting the Test set results
    y_pred = classifier.predict(X_test)
    y_pred = (y_pred > 0.5)
    
    
    from sklearn.model_selection import GridSearchCV
    
    array = np.array([[month,field,sat,gre,awa,toefl,ielts,work,paper,loan,international,grade,university]])
    college  = str(array[:,12])
    college = college.replace('[', '')
    college = college.replace(']', '')
    college = college.replace("'", "")
    
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
    
    
    word  = str(array[:,12])
    word = word.replace('[', '')
    word = word.replace(']', '')
    word = word.replace("'", "")
    ans = classes2[word]
    array[:,12] = ans
    
    
    OldValue = float(array[:,2])
    NewMin = 130
    OldMin =400
    OldMax =1600
    NewMax = 340
    NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    array[:,2] = NewValue
    
    
    
    
    new_prediction = classifier.predict(sc.transform(array))
    new_prediction = (new_prediction > 0.5)
    print(new_prediction)
    
    
    y_test = (y_test > 0.5)
    	# Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    if(new_prediction[0] ==True):
        response = "You are more likely to be accepted into " + college
    else:
        response = "You are more likely to be rejected into " + college
    return response


def increase(month,field,sat,gre,awa,toefl,ielts,work,paper,loan,international,grade,university):
# Importing the libraries
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import re
    	# Importing the dataset
    dataset = pd.read_csv('data_set.csv',error_bad_lines=False, encoding = "ISO-8859-1")
    
    	  
    X = dataset.iloc[:, 1:14].values
    y = dataset.iloc[:, 15].values
    
    X[:,0] = np.random.randint(1, 12, size=len(X[:,0]))
    
    
    	# Encoding categorical data
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    classes = dict(zip(labelencoder_X_1.classes_, labelencoder_X_1.transform(labelencoder_X_1.classes_)))
    
    
    
    
    labelencoder_X_2 = LabelEncoder()
    X[:, 12] = labelencoder_X_2.fit_transform(X[:, 12])
    classes2 = dict(zip(labelencoder_X_2.classes_, labelencoder_X_2.transform(labelencoder_X_2.classes_)))
    
    
    labelencoder_y_1 = LabelEncoder()
    y = labelencoder_y_1.fit_transform(y)
    y = y.astype(int)
    
    	#Missing Data
    from sklearn.preprocessing import Imputer
    imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(X[:, :])
    X[:, :] = imputer.transform(X[:, :])
    X = X.astype(float)
    import statsmodels.formula.api as sm
    #X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1) # create a constant in the regression model
    
    X_opt = X[:,[0, 1, 2, 3, 4,5,6,7,8,9,10,11,12]] # creates a new set for optimal X values
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
    regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)
    
    X_opt = X[:,[0, 1, 2, 3, 4,5,6,7,8,9,10,11]] # creates a new set for optimal X values
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
    regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)
    
    X_opt = X[:,[0, 1, 2, 3, 4,6,7,8,9,10,11]] # creates a new set for optimal X values
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
    regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)
    
    X_opt = X[:,[0, 1, 2, 3, 4,6,7,9,10,11]] # creates a new set for optimal X values
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
    regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)
    
    X_opt = X[:,[1,6]] # creates a new set for optimal X values
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # fits the X values into the backward eliminator
    regressor_OLS.summary() # shows all of the matrixes of the data (the lower the p value the more singnificant the varibale is to the data set)
    
    if(sat<1200):
        response = "You should retake your SAT to get a better score"
    if(work ==0):
        response2 = " You should consider working as an intern"
    
        return response + "\n" + response2
    
