
#If you get an import error with MLPClassifier, try
""" !conda update scikit-learn """

from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier

import numpy as np
import pandas as pd


# 
# try an iris classifier...
#



print("+++ Start of iris example +++\n")
# For Pandas's read_csv, use header=0 when you know row 0 is a header row
# df here is a "dataframe":
df = pd.read_csv('app\quiz.csv', header=0)    # read the file
df.head()                                 # first few lines

# One important feature is the conversion from string to numeric datatypes!
# For _input_ features, numpy and scikit-learn need numeric datatypes
# You can define a transformation function, to help out...
def transform_transportation(s):
    """ from string to number
    		Transportation:
          Walking -> 0, Skating -> 1, Boarding -> 2, Biking -> 3, Scooter -> 4
    """
    T = { 'unknown':-1, 'Walking':0, 'Skating':1, 'Boarding':2, 'Biking': 3, 'Scooter':4 }
    
    return T[s]
  
def transform_hangout(s):
    """ from string to number
        Hangout Spot:
        	Motley -> 0, The Coop -> 1, The Hub -> 2, The Cafe -> 3, Milk and Honey -> 4, Jay's Place -> 5 
    """
    HS = { 'unknown':-1, 'Motley':0, 'The Coop':1, 'The Hub':2, 'The Cafe': 3, 'Milk and Honey':4, "Jay's Place":5 }
    
    return HS[s]
  
def transform_sushi(s):
    """ from string to binary
            likes sushi -> 1, dislikes sushi -> 0
    """
    SU = { 'Yes':1, 'No':0 }
    return SU[s]

def transform_pepo_melo(s):
    """ from string to binary
            has been to pepo melo -> 1, has not been to pepo melo -> 0
    """
    PM = { 'Yes':1, 'No':0 }
    return PM[s]

def transform_dessert(s):
    """ from string to binary
            fruit -> 1, ice cream -> 0
    """
    SU = { 'Fruit':1, 'Ice cream':0 }
    return SU[s]

def transform_answer(s):
    """ from string to binary
            fruit -> 1, ice cream -> 0
    """
    ans = {'unknown':-1, 'Frank':0 , 'Frary':1, 'Oldenborg':2, 'Collins':3, 'Mallot':4, 'The Hoch':5, 'McConnell':6}
    return ans[s]
# 
# this applies the function transform to a whole column
#
df = df.drop('cookies', axis=1)

functions = [transform_transportation, transform_hangout, transform_sushi, transform_pepo_melo, transform_dessert, transform_answer]
columns = ['transportation', 'hangout', 'sushi', 'pepo_melo', 'dessert', 'answer']
for i in range(6):
    df[columns[i]]=df[columns[i]].map(functions[i])

print("+++ Converting to numpy arrays... +++")
# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_data_complete = df.iloc[:,0:9].values         # iloc == "integer locations" of rows/cols
y_data_complete = df[ 'answer' ].values       # individually addressable columns (by name)

X_unknown = X_data_complete[:6,:]
y_unknown = y_data_complete[:6]

X_known = X_data_complete[6:,:]
y_known = y_data_complete[6:]

#
# we can scramble the remaining data if we want to (we do)
# 
KNOWN_SIZE = len(y_known)
indices = np.random.permutation(KNOWN_SIZE)  # this scrambles the data each time
X_known = X_known[indices]
y_known = y_known[indices]

#
# from the known data, create training and testing datasets
#
TRAIN_FRACTION = 0.85
TRAIN_SIZE = int(TRAIN_FRACTION*KNOWN_SIZE)
TEST_SIZE = KNOWN_SIZE - TRAIN_SIZE   # not really needed, but...
X_train = X_known[:TRAIN_SIZE]
y_train = y_known[:TRAIN_SIZE]

X_test = X_known[TRAIN_SIZE:]
y_test = y_known[TRAIN_SIZE:]

#
# it's important to keep the input values in the 0-to-1 or -1-to-1 range
#    This is done through the "StandardScaler" in scikit-learn
# 
USE_SCALER = True
if USE_SCALER == True:
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)   # Fit only to the training dataframe
    # now, rescale inputs -- both testing and training
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_unknown = scaler.transform(X_unknown)

# scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html 
#

mlp = MLPClassifier(hidden_layer_sizes=(10,10,), max_iter=400, alpha=1e-4,
                    solver='sgd', verbose=True, shuffle=True, early_stopping = False, tol=1e-4, 
                    random_state=None, # reproduceability
                    learning_rate_init=.1, learning_rate = 'adaptive')

print("\n\n++++++++++  TRAINING  +++++++++++++++\n\n")
mlp.fit(X_train, y_train)


print("\n\n++++++++++++  TESTING  +++++++++++++\n\n")
print("Training set score: %f" % mlp.score(X_train, y_train))
print("Test set score: %f" % mlp.score(X_test, y_test))

# let's see the coefficients -- the nnet weights!
# CS = [coef.shape for coef in mlp.coefs_]
# print(CS)

# predictions:
predictions = mlp.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print("\nConfusion matrix:")
print(confusion_matrix(y_test,predictions))

print("\nClassification report")
print(classification_report(y_test,predictions))

# unknown data rows...
#
unknown_predictions = mlp.predict(X_unknown)
print("Unknown predictions:")
print("  Correct values:  ['Collins', 'Collins', 'Mallot', 'McConnell', 'The Hoch', 'The Hoch']")
print("  Our predictions: ", unknown_predictions)

#
# You can use the function you've trained on any input data--
# here's a function to demonstrate how!
#
# Call it with ownData() to use our example, or input your own numbers
#

def ownData(a,b,c,e,f,g,h,i,j):
    L = [a,b,c,e,f,g,h,i,j]
    row = np.array(L)  # makes an array-row
    row = row.reshape(1,9)   # makes an array of array-row
    if USE_SCALER == True:
        row = scaler.transform(row)
    d = { -1:'unknown',0:'Frank',1:'Frary',2:'Oldenborg',3:'Collins',4:'Mallot',5:'The Hoch', 6:'McConnell'} 

    return d[int(mlp.predict(row))]
