# Artificial Neural Network 


# Part 1 - Data Preprocessing

# Scale Data: NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin

'''
                         __________
                      .~#########%%;~.
                     /############%%;`\
                    /######/~\/~\%%;,;,\
                   |#######\    /;;;;.,.|
                   |#########\/%;;;;;.,.|
          XX       |##/~~\####%;;;/~~\;,|       XX
        XX..X      |#|  o  \##%;/  o  |.|      X..XX
      XX.....X     |##\____/##%;\____/.,|     X.....XX
 XXXXX.....XX      \#########/\;;;;;;,, /      XX.....XXXXX
X |......XX%,.@      \######/%;\;;;;, /      @#%,XX......| X
X |.....X  @#%,.@     |######%%;;;;,.|     @#%,.@  X.....| X
X  \...X     @#%,.@   |# # # % ; ; ;,|   @#%,.@     X.../  X
 X# \.X        @#%,.@                  @#%,.@        X./  #
  ##  X          @#%,.@              @#%,.@          X   #
, "# #X            @#%,.@          @#%,.@            X ##
   `###X             @#%,.@      @#%,.@             ####'
  . ' ###              @#%.,@  @#%,.@              ###`"
    . ";"                @#%.@#%,.@                ;"` ' .
      '                    @#%,.@                   ,.
      ` ,                @#%,.@  @@                `
                          @@@  @@@  
'''


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


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.5)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

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


array = np.array([['Dec', 'Others', 1600, 111, 10, 19, 101, 1, 1, 1, 1, 100, 'Arizona State University']])

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

word  = str(array[:,0])
word = word.replace('[', '')
word = word.replace(']', '')
word = word.replace("'", "")
ans = month[word]
array[:,0] = ans


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




# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu', input_dim = 15))
    classifier.add(Dense(units = 6, kernel_initializer = 'glorot_uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = 1)
mean = accuracies.mean()
variance = accuracies.std()

# Improving the ANN
# Dropout Regularization to reduce overfitting if needed

# Tuning the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 15))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier)
parameters = {'batch_size': [25, 32],
              'epochs': [100, 500],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_



'''
                                       [/~~~~~~l'
                                [%};       `rl
                               Odq            O
                               B,             8i
                              vBc            x n
                             xB xn          xW W
                             xW wn '     '' xW xn
                             x$ nxBZB   x`B$x$ %n
                             xBO.xZJB   xOhB x Z.
                              x`  hZ "in `n   B
                              +B[/; '%n$1 ;l "0
                               `@$  0}.)0 "a$a
                                xB;(O'''[OU>@n
                                xWlw0dqqd8*]a,
                                 +W 000000 w,
                                  +@'    '0,
                                    'BB@B'
                                    OJi"UO.
                                   "`U``h`i
                                    B%$%$`i                              
                          '''''~}))}~W  w0Ma~l'
                     '~~~C`}0)}~/((OO};])Ok*/z0~~~'''''
                  ':(8''''OOOOO'! f8B/~~(8fQQfOO/~88888#~''
                 "8Qf``8qd`#`W#BW'8#`.   80q#BBd,Ot`\``Z t8O
                wB0c B 0, B@' +0BBBBB   w0BB@W0 v BBW'Bc0B0 xW
               xW '+,B   wB 0Ww'  +0' +,x00, '0Bn'nB   w0    x
              xB  B  B  Bn'B'+cB0@BW0   +'BBBBBBB0wBn '@B    x
              xBi B"U`n %n`ZBh$OU```     ```."OOUB" $O ZBU. xZ
               hU.`xn hO`h hBBBBBBB$   . BBBBBB$%O.BxBOn Z. xn
               xJi Bn  B h`OO ```.OO      i""iO"OOUO.x`  xZ xn
               x!  n  +8c?0(8B$OOa}.      0@%$BBB0]8kW0  xW xn
               x!  n  "`zX'f8```8QO1     '1?````t'/8pd n xW xn
               x!  n  _OvM8\0~~~00}`'    `)rOOOO%B*0`>Oc xW xn
               x!  n  x`;CB#O1['1[  Bi "%O  \t`\t`>'O%0x xW +c
              xW   n  +B'    +000,wB'000@0W @BBBBBBBB,'n xW  x
              xW  ',   BwB0WBBW' xB0B@BB@0wW +00000 'w0B xW  x
              xW  B   x'+0B'BBB wB 'B' 'wB'w  BwW'BBcv B +c  x
              x$  B    Bi"  ``.O$%ZJB   xBh$B$ `h`Z`."B.  x` x
              x$  B    BhZ``B."B`.xBB%$$%Bi `hBi"OOBhZ`i  x` x
              UJ  B    BU`O%`xZ O%BB` ..xBBBOi`$iBBBU.On  x`  n
              W!  B    BhBB OZJOB};zB[11w8  )@Ovc Q`[%B   x8  n
              W!  0    0?)0"0$%0.  _B)}}h8    0Ww hOB}`n  x8  n
              W!   n    c''a BW    zB   x8    'BZO?`>[~.  w8  n
              W!   c     ``. )0    _B(//k8    0}.`MOa}   wB`  Ji
             'n xW xn             +cB,  x',              @n   +@'
             0BWwB'xn               BBBBB                Bn wB  B
              wBv@Bn     ''B0@W'   vBv,,x0    'B00@'     xB'xB'w0
             B. n"`    OZ` O   `n   B. "%`  xZ   `$ h$    xBi" x
            "B.x"U    "B."`     Ji%B`ZhhB`$ixn     `  B    Bi n B
            xO"UU.    %` J       h"U     `ix.        J`n   Bi n `i
           xW v,n     W z!      a}  O}))qB  Mi        O,    W!hn c
           a} n ,     W _!      n  "0    `l v,       [0     @W+M xn
           * O.x      h'_Q      ;l/] '  l' J,       xZ      xW xO+M
          x! B x       0%B'    'Oa$i'\. _8 x~l      U.       h xB x
         xB xBxB        +@'    Bn xW '  ' xB x     B         x  Bn n
         wBxW0xn          B    0W  +B     B,xB    x'          B,0n xn
        ', nx B,       ''vB     +@   c  w0'w,    B,B          @W Bn+@
       "`."JUOn       %`$%O O   Oi hOOUJOB`.    x  O$%`J      x$ hn  B
       x` xxnB       O.    " $UBZ`$i %ZhnOOOOOO%n "B`. xO      x`xn  Bi
      x$  xix        BJ`  O"Bn B   x`hnxZBn  x x$OB   O B       B%B   $i
      +M8 xlv       xB>` xB` W `r~}>OZ;]J'`)}. a}`%~  B 0       `/k  ''w
      :C8rCU.       +B+  x`  )M'  [a`.   01' 'O,  @'  0a         *kl[0W\
    [~W'0l'n         0i  +'    `rC}.      `)C0`   %0   n         Ja+q~0cO1
   O}O)@'wWc          $'  B                       B   ',         ;:C':kB`~
  w w BnBxWxn         x',  n                      B   B           xc 'nB@'xW
 wn'B W B wxn         xB,  n                     Bn   0           x0c0vc0 n+'
'cv'n0W B wxn         +B   c                     Bn  x            xxB nxn +c0n
O. `OBn B.x.           B$i x                    "B  xn            x h nOJ   `
    nBxZBZh            Bn  x                    xB  xn            $%BhnBx
    $BxnBnx             $U  B                   xO. n             nxBxnB%
    nBhZBJU             @Z~ 0                  xB~ ',             JUBhZBx
    *Bk*Bv,             x$~  n                 xW  B              +cBk*Bk
    cBxW0.               h'  Ji                %W  B                0wnBv
     `]`                 xB! xn               'B} "`                 `;`.
                          B   x               BB, x
                          0B  x               BB, x
                           B'  B             xB   n
                           hO  `i            x`   n
                           xO   Ji          xB`  B
                            B.  xn          xB`  B
                            B!   +          Br   0
                            `*    c        [,   x
                             $1   xn      xB    x
                            'B}'' +c      ac  ' +O
                            B x  x +B    B  v00   n
                            B +' xW B    B  W  n  n
                           +B  B wxW0    0n xn'n 'n
                             BZ``JBn      ``%n`JOB
                           "```BZ`xn     OZ`h$B`h
                           xOi"   Zh     `OO   %B
                           Z W\B"Uxn      xZn B w
                           n W!0v,xn      x*n Bv@
                           nOW! . xn      x*;C`nx
                           nB)M'  v,      x*  Ocw
                           nB x'  n       xB  BxB
                           c nx'  n       xB  BxB
                           x n B, n       xB x wB
                           x $iB. n       xZ x nx
                            OxnB. n       %Z x nx
                            BxnB. n       $i x nx
                            BxWB! n       *  hB w
                            `UhB! n       *  xBxZ
                             nxB! n       * "%Bxn
                             cvB! c       * xB`xn
                             xnB  xn     'B,xB xn
                             xnBn xn     B  xn B,
                             +c0B,+c     B  xn n
                              x Z. x     Bi xn n
                              xOZ.  B    Bi n B.
                               B    B    Bi n B
                               B     n  x8  MiB
                               B1'' ',  x8  wnB
                               `W\`;8   x8~~BnB
                               O0M1[Bn  xB''h$B
                               Bc'v, n  x'  BB
                               B +BBB   x xB  B
                             v,'cwwn0n  x BW   0c'
                           UB%ZBhZ J n x$ Z.hZBi"`$
                           `OJ% Bn x`n  x`nx`h`Zh`U
                            `.  n   O.   O. h$`hn`
'''
