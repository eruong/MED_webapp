# Authors: Melody Chang, Ethan Ong, Dylan Hou
# File that builds model from survey data to predict favorite dining hall
# Model Selection:


import numpy as np            
import pandas as pd

from sklearn import tree      # for decision trees
from sklearn import ensemble  # for random forests

try: # different imports for different versions of scikit-learn
    from sklearn.model_selection import cross_val_score   # simpler cv this week
except ImportError:
    try:
        from sklearn.cross_validation import cross_val_score
    except:
        print("No cross_val_score!")
        

#
# Here are the correct answers to the csv's "unknown" flowers
#
answers = [ 'Collins', 'Collins', 'Mallot', 'McConnell', 'The Hoch', 'The Hoch']



# print("+++ Start of pandas' datahandling +++\n")

# df is a "dataframe":
df = pd.read_csv('quiz.csv', header=0)   # read the file w/header row #0

# Now, let's take a look at a bit of the dataframe, df:
# df.head()                                 # first five lines
# df.info()                                 # column details

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


# print("\n+++ End of pandas +++\n")

# print("+++ Start of numpy/scikit-learn +++\n")

# print("     +++++ Decision Trees +++++\n\n")

# Data needs to be in numpy arrays - these next two lines convert to numpy arrays
X_all = df.drop('answer', axis=1).values         # iloc == "integer locations" of rows/cols  
y_all = df['answer'].values      # individually addressable columns (by name)

X_unlabeled = X_all[:6,:]  # the "unknown" flower species (see above!)
y_unlabeled = y_all[:6]    # these are "unknown"

X_labeled = X_all[6:,:]  # make the 10 into 0 to keep all of the data
y_labeled = y_all[6:]    # same for this line

#
# we can scramble the data - but only the labeled data!
# 
indices = np.random.permutation(len(X_labeled))  # this scrambles the data each time
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

#
# Notice that, here, we will _only_ use cross-validation for model-buidling
#   (We won't include a separate X_train X_test split.)
#

X_train = X_data_full
y_train = y_data_full

#
# some labels to make the graphical trees more readable...
#
# print("Some labels for the graphical tree:")
feature_names = df.drop('answer', axis=1).columns.values.tolist()
target_names = ['Frank', 'Frary', 'Oldenborg', 'Collins', 'Mallot', 'The Hoch', 'McConnell']

#
# show the creation of three tree files (at three max_depths)
#
# for max_depth in [3,5,10]:
#     # the DT classifier
#     dtree = tree.DecisionTreeClassifier(max_depth=max_depth)

#     # train it (build the tree)
#     dtree = dtree.fit(X_train, y_train) 

#     # write out the dtree to tree.dot (or another filename of your choosing...)
#     filename = 'tree' + str(max_depth) + '.dot'
#     tree.export_graphviz(dtree, out_file=filename,   # the filename constructed above...!
#                             feature_names=feature_names,  filled=True, 
#                             rotate=False, # True for Left-to-Right; False for Up-Down
#                             class_names=target_names, 
#                             leaves_parallel=True )  # lots of options!
#     #
#     # Visualize the resulting graphs (the trees) at www.webgraphviz.com
#     #
#     print("Wrote the file", filename)  
    #



#
# cross-validation and scoring to determine parameter: max_depth
# 
for max_depth in range(1,12):
    # create our classifier
    dtree = tree.DecisionTreeClassifier(max_depth=max_depth)
    #
    # cross-validate to tune our model (this week, all-at-once)
    #
    scores = cross_val_score(dtree, X_train, y_train, cv=5)
    average_cv_score = scores.mean()
    print("      Scores:", scores)
    print("For depth=", max_depth, "average CV score = ", average_cv_score)


print("\n\n")
print("     +++++ Random Forests +++++\n\n")

#
# The data is already in good shape -- let's start from the original dataframe:
#
X_all = df.drop('answer', axis=1).values        # iloc == "integer locations" of rows/cols
y_all = df[ 'answer' ].values      # individually addressable columns (by name)

#
# Labeled and unlabeled data...
#

X_unlabeled = X_all[:6,:]  # the "unknown" flower species (see above!)
y_unlabeled = y_all[:6]    # these are "unknown"

X_labeled = X_all[6:,:]  # just the input features, X, that HAVE output labels
y_labeled = y_all[6:]    # here are the output labels, y, for X_labeled

#
# we can scramble the data - but only the labeled data!
# 
indices = np.random.permutation(len(X_labeled))  # this scrambles the data each time
X_data_full = X_labeled[indices]
y_data_full = y_labeled[indices]

X_train = X_data_full
y_train = y_data_full

#
# Again, we will use cross-validation to determine the Random Forest's two hyperparameters:
#   + max_depth
#   + n_estimators
#
# (We will not have an X_train vs. X_test split in addition.)
#

#
# Lab task!  Your goal:
#   + loop over a number of values of max_depth (m)
#   + loop over different numbers of trees/n_estimators (n)
#   -> to find a pair of values that results in a good average CV score
#
# use the decision-tree code above as a template for this...
#


# here is a _single_ example call to build a RF:
# You need to "loopify" the code...
greatest_mean = 0
d = 0
t = 0
for m in range(1,7):
    for n in range(50, 300, 100):
        rforest = ensemble.RandomForestClassifier(max_depth=m, n_estimators=n)

        # an example call to run 5x cross-validation on the labeled data
        scores = cross_val_score(rforest, X_train, y_train, cv=5)
        # print("CV scores:", scores)
        # print("Parameters: max_depth =", m,"; n_estimators =", n, "\n CV scores' average:", scores.mean())
        if scores.mean() > greatest_mean:
            d = m
            t = n
            greatest_mean = scores.mean()

print('the greatest CV score is:', d, t, '-', greatest_mean)

#
# now, train the model with ALL of the training data...  and predict the labels of the test set
#


# these next lines is where the full training data is used for the model
MAX_DEPTH = d
NUM_TREES = t
print()
print("Using MAX_DEPTH=", MAX_DEPTH, "and NUM_TREES=", NUM_TREES)
rforest = ensemble.RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=NUM_TREES)
rforest = rforest.fit(X_train, y_train) 

# here are some examples, printed out:
print("Random-forest predictions:\n")
predicted_nums = rforest.predict(X_unlabeled)
answer_labels = answers  # because we know the answers, above!

# change predictions back into flower names
predicted_labels = [target_names[x] for x in predicted_nums]

#
# formatted printing again (see above for reference link)
#
s = "{0:<11} | {1:<11}".format("Predicted","Answer")
#  arg0: left-aligned, 11 spaces, string, arg1: ditto
print(s)
s = "{0:<11} | {1:<11}".format("-------","-------")
print(s)
# the table...
for p, a in zip( predicted_labels, answer_labels ):
    s = "{0:<11} | {1:<11}".format(p,a)
    print(s)

#
# feature importances
#
print("\nrforest.feature_importances_ are\n      ", rforest.feature_importances_) 
print("Order:", feature_names[0:9])



