

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score



os.chdir("C:/Users/rishabh/Desktop/TCS_Project")





# Reading the excel file



master_quote = pd.read_excel("master_quotes-09RNH_fn.xlsx", encoding='utf-8')


# Dropping all the columns with NaN values
master_quote = master_quote.dropna(axis=1, how = 'all', inplace = False)

# Summing up all missing values from each column in the dataframe

missing_val_quote = pd.DataFrame(master_quote.isna().sum())

missing_val_quote = missing_val_quote.reset_index()

missing_val_quote = missing_val_quote.rename(columns = {'index': 'columns', 0: 'missing_val' })


# Finding the Percentage of the missing values in each column

missing_val_quote['missing_val'] = (missing_val_quote['missing_val']/len(master_quote))*100


# sorting the missing value percentage in descending order


missing_val_quote = missing_val_quote.sort_values('missing_val', ascending = False).reset_index(drop = True)

# Using for loop to drop the values having more than 60% null values

count = 0

for column in missing_val_quote['columns']:
    if count <= 28:
        print(column)   
        master_quote.drop(column,axis = 1,inplace = True)
        count += 1
     
    else:
        break

master_quote.drop("ROW_ID",axis = 1,inplace = True)

# Imputing the missing values with mode


# In[26]:


for column in list(master_quote.columns):
    master_quote[column].fillna(master_quote[column].mode()[0], inplace = True)

# checking the percentage of "Lost" in STAT_CD column

master_quote_label = pd.DataFrame(master_quote.iloc[:,-1])

# (len(master_quote_label.loc[master_quote_label["STAT_CD"]=="Lost"])/len(master_quote_label))*100
# 86.521


# Dropping the columns still having null values

df_no_date = master_quote_features.drop(["DUE_DT","EFF_END_DT","EFF_START_DT","QUOTE_EXCH_DT"],axis = 1)

# Encoding the categorical values

df_no_date = pd.get_dummies(df_no_date)

# Joining the features with Target Column
df_no_date = df_no_date.join(master_quote_label.iloc[:,-1])


# Splitting the data into test and train


Xd_train, Xd_test, yd_train, yd_test = train_test_split(df_no_date.drop(['STAT_CD'], axis=1), df_no_date['STAT_CD'],test_size = .4,random_state=12)


# In[35]:


sm = SMOTE(random_state=12, ratio = 1.0)


# Fitting the train data into the SMOTE classifier to get the the equal proportion of target variable class

Xd_train_res, yd_train_res = sm.fit_sample(Xd_train, yd_train)


# checking the ratio of "Lost" and "Order Placed"

# len(pd.DataFrame(yd_train_res).loc[pd.DataFrame(yd_train_res)[0]== "Lost"])/len(pd.DataFrame(yd_train_res).loc[pd.DataFrame(yd_train_res)[0] != "Lost"])


# Using the Random Forest Classifier, fitting the model into the training dataset

clf_rf = RandomForestClassifier(n_estimators = 100, random_state = 12).fit(Xd_train_res, yd_train_res)


# Predict the test data

test_prediction = clf_rf.predict(Xd_test)

print(accuracy_score(yd_test, test_prediction))


# Creating the matrix to check the accuracy

CM = pd.crosstab(yd_test, test_prediction)


TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]



#accuracy_score(y_test, y_pred)*100
print(((TP+TN)*100)/(TP+TN+FP+FN))


#False Negative rate 
print((FN*100)/(FN+TP))


# # Model Serialization

import requests,json
from sklearn.externals import joblib


##
##
##pickle.dump(clf_rf,open("clf_rf.pkl","wb"))
##
##
### In[50]:
##
##
##my_random_forest = pickle.load(open("clf_rf.pkl","rb"))
##
##


##
##url = "http://localhost:12345/api"
##data = json.dumps({})
##

joblib.dump(clf_rf,"model.pkl")

rf = joblib.load("model.pkl")

print(accuracy_score(yd_test, rf.predict(Xd_test)))

model_columns = list(Xd_train.columns)

joblib.dump(model_columns,"model_columns.pkl")
