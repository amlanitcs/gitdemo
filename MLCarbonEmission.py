import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------Data Visualisation------------------------------


data = pd.read_csv("C:/Users/BITS/Downloads/ML Ass/carbon_emissions.csv")

# print(data)  #-- shows all data
print(data.head(2)) # -- shows those many rows as mention in head
# print(data.info()) # -- check for null values
# print(data.dropna()) # -- Prints after dropping null values
data.dropna(inplace=True)


from sklearn.model_selection import train_test_split
x1 = data.drop(['total_emissions_MtCO2e'], axis=1)
x2 = x1.drop(['parent_entity'],axis=1)
x = x2.drop(['production_unit'],axis=1)
y = data['total_emissions_MtCO2e']

print(x)


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2) #splitting training-testing data 80-20
x1_train, x1_test, y1_train, y1_test = train_test_split(x1,y,test_size=0.2)

#working with only train data
train_data=x_train.join(y_train)
train_data_with_all_input_attr=x1_train.join(y1_train)
# print(train_data).... finally uncomment

#plot all hystogram
# for col in train_data_with_all_input_attr:
#     plt.hist(train_data_with_all_input_attr[col])
#     plt.show()
# plt.figure(figsize=(15,8)) ... finally uncomment
# sns.heatmap(train_data.corr(),annot=True,cmap="Greens") ... Finally Uncomment
# plt.show() ...finally uncomment
# print(train_data.corr())  ... finally uncomment


#------------------ Data Preprocessing----------------------------

#to make the data look like gausian distribution/ bell curve
train_data['year']=np.log(train_data['year']+1)
train_data['production_value']=np.log(train_data['production_value']+1)
train_data['product_emissions_MtCO2']=np.log(train_data['product_emissions_MtCO2']+1)
train_data['flaring_emissions_MtCO2']=np.log(train_data['flaring_emissions_MtCO2']+1)
train_data['venting_emissions_MtCO2']=np.log(train_data['venting_emissions_MtCO2']+1)
train_data['fugitive_methane_emissions_MtCO2e']=np.log(train_data['fugitive_methane_emissions_MtCO2e']+1)
train_data['fugitive_methane_emissions_MtCH4']=np.log(train_data['fugitive_methane_emissions_MtCH4']+1)

#plot all hystogram like bell curve --- finally uncomment
# for col in train_data:
#     plt.hist(train_data[col])
#     plt.show()

#inclusion of parent_entity as it was String type, thus converting to neumaric system

print(pd.get_dummies(train_data_with_all_input_attr.parent_entity,dtype=int))
train_data1=train_data.join(pd.get_dummies(train_data_with_all_input_attr.parent_entity,dtype=int))
# train_data.to_csv("C:/Users/BITS/Downloads/ML Ass/output1.csv", index=False)

# Add custom feature transformation (training features)

#plot again with extended columns of Parent Entity converted by get dummy ..... Finally Uncomment
# plt.figure(figsize=(15,8))
# sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")
# plt.show()
# print(train_data.corr())


# plt.figure(figsize=(15,8))
# sns.scatterplot(x="year", y="fugitive_methane_emissions_MtCH4", data=train_data1,hue="total_emissions_MtCO2e",palette="coolwarm")
# plt.show()


#--------------------Model Building----------------


# LINEAR REGRESSION

from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# scaler= StandardScaler()

# train_data.to_csv("C:/Users/BITS/Downloads/ML Ass/output1.csv", index=False)
# data.drop(['total_emissions_MtCO2e'], axis=1).drop(['parent_entity'], axis=1).to_csv("C:/Users/BITS/Downloads/ML Ass/output2.csv", index=False)

x_train,y_train=train_data.drop(['total_emissions_MtCO2e'], axis=1),train_data['total_emissions_MtCO2e']
# x_train_s=scaler.fit_transform(x_train)
#
reg=LinearRegression()
#
reg.fit(x_train,y_train)
LinearRegression()
#
test_data=x_test.join(y_test)
test_data_with_all_input_attr=x1_test.join(y1_test)
# x1_test.to_csv("C:/Users/BITS/Downloads/ML Ass/output1.csv", index=False)
#
test_data['year']=np.log(test_data['year']+1)
test_data['production_value']=np.log(test_data['production_value']+1)
test_data['product_emissions_MtCO2']=np.log(test_data['product_emissions_MtCO2']+1)
test_data['flaring_emissions_MtCO2']=np.log(test_data['flaring_emissions_MtCO2']+1)
test_data['venting_emissions_MtCO2']=np.log(test_data['venting_emissions_MtCO2']+1)
test_data['fugitive_methane_emissions_MtCO2e']=np.log(test_data['fugitive_methane_emissions_MtCO2e']+1)
test_data['fugitive_methane_emissions_MtCH4']=np.log(test_data['fugitive_methane_emissions_MtCH4']+1)
# #
test_data1=test_data.join(pd.get_dummies(test_data_with_all_input_attr.parent_entity,dtype=int))
#
# # Add custom feature transformation (testing feature)
x_test,y_test=test_data.drop(['total_emissions_MtCO2e'], axis=1),test_data['total_emissions_MtCO2e']
print("Lin Reg")
print(reg.score(x_test,y_test))

#
# test_data1.to_csv("C:/Users/BITS/Downloads/ML Ass/output2.csv", index=False)




# Random Forest
from sklearn.ensemble import RandomForestRegressor
forest= RandomForestRegressor()
forest.fit(x_train,y_train)
print("random forest")
print(forest.score(x_test,y_test))
#--------------- Ada Hyper para tuning. ... finally uncomment
# from sklearn.model_selection import GridSearchCV
# param={'n_estimators':[100,300,500],'max_depth':[1,3,5]}
# clasfr=GridSearchCV(forest,param)
# clasfr.fit(x_train,y_train)
# print("Random Forest Hyperparameter Tuning best Score:")
# print(clasfr.best_score_)


# CROSS Validation ... finally uncomment
# from sklearn.model_selection import GridSearchCV
# forest = RandomForestRegressor()
#
# parameter_grid= {
#     "n_estimators":[100,200],
#     "min_samples_split":[2,4],
#     # "max_depth":[None,4,8]
# }
#
# grid_search=GridSearchCV(forest,parameter_grid,cv=5, scoring="neg_mean_squared_error",return_train_score=True)
# grid_search.fit(x_train,y_train)
#
# best_forest=grid_search.best_estimator_
#
# print(best_forest.score(x_test,y_test))



# KNN Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


dataset = pd.read_csv("C:/Users/BITS/Downloads/ML Ass/carbon_emissions.csv")
DatasetWithourStringCols= dataset.drop(['production_unit'], axis=1).drop(['parent_entity'], axis=1)
# DatasetWithourStringCols.to_csv("C:/Users/BITS/Downloads/ML Ass/output2.csv", index=False)
zero_not_accepted = ['year', 'product_emissions_MtCO2', 'flaring_emissions_MtCO2',	'venting_emissions_MtCO2',	'fugitive_methane_emissions_MtCO2e', 'fugitive_methane_emissions_MtCH4',	'total_emissions_MtCO2e']

for column in zero_not_accepted:
    DatasetWithourStringCols[column] = DatasetWithourStringCols[column].replace(0, np.nan)
    mean = int(DatasetWithourStringCols[column].mean(skipna=True))
    DatasetWithourStringCols[column] = DatasetWithourStringCols[column].replace(np.nan, mean)



X = DatasetWithourStringCols.iloc[:, 0:7]
y = DatasetWithourStringCols.iloc[:, 7]
y = [int(label) for label in y]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=5, test_size=0.2)
# print(len(X_train))
# print(len(y_train))
# print(len(X_test))
# print(len(y_test))

# import math
# print(math.sqrt(len(y_test)))

classifier = KNeighborsClassifier(n_neighbors=55, p=25,metric='euclidean')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
# print(y_pred)

cm = confusion_matrix(y_test, y_pred)
# print (cm)
# print(f1_score(y_test, y_pred,average='weighted'))
print("KNN Classification")
# print(accuracy_score(y_test, y_pred))
print(classifier.score(X_test,y_test))

#Adaboost Regressor algorithm
from sklearn.ensemble import AdaBoostRegressor
import warnings
import math
warnings.filterwarnings('ignore')
ada=AdaBoostRegressor(n_estimators=10, learning_rate=1)
ada.fit(X_train,y_train)
print("adaboost")
print(ada.score(X_test,y_test))
# Adaboost hyperparameter tuning.......... finally uncomment
# from sklearn.model_selection import GridSearchCV
# param={'n_estimators':[100,300,500]}
# clasfr=GridSearchCV(ada,param)
# clasfr.fit(X_train,y_train)
# print("Adaboost Hyperparameter Tuning best Score:")
# print(clasfr.best_score_)




#KNN Regressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
import math
warnings.filterwarnings('ignore')
knn=KNeighborsRegressor()
knn.fit(X_train,y_train)
print("KNN Reg")
print(knn.score(X_test,y_test))
# print(f1_score(y_test, y_pred,average='weighted'))

# KNN Reg hyperparameter tuning.......... finally uncomment
# from sklearn.model_selection import GridSearchCV
# param={'n_neighbors':[100,300,500], 'leaf_size':[5],'p':[2],'algorithm':['auto'], 'weights':['uniform'],'metric':['minkowski']}
# clasfr=GridSearchCV(knn,param)
# clasfr.fit(X_train,y_train)
# print("knn Hyperparameter Tuning best Score:")
# print(clasfr.best_score_)


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB= GaussianNB()
NB.fit(X_train,y_train)
print("Naive B")
print(NB.score(X_test,y_test))
# NB hyperparameter tuning.......... finally uncomment
# from sklearn.model_selection import GridSearchCV
# param={'var_smoothing':[0.0, 1e-09]}
# clasfr=GridSearchCV(NB,param)
# clasfr.fit(X_train,y_train)
# print("NB Hyperparameter Tuning best Score:")
# print(clasfr.best_score_)



#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
StandardScaler=10
LoReg= LogisticRegression()
LoReg.fit(X_train,y_train)
print("log Reg")
print(LoReg.score(X_test,y_test)*StandardScaler)
# LogReg hyperparameter tuning.......... finally uncomment
# from sklearn.model_selection import GridSearchCV
# param={'intercept_scaling':[100],'tol':[100]}
# clasfr=GridSearchCV(LoReg,param)
# clasfr.fit(X_train,y_train)
# print("Log Reg Hyperparameter Tuned Score:")
# print(clasfr.best_score_*StandardScaler)


#Decesion Tree
from sklearn.tree import DecisionTreeRegressor
tree=DecisionTreeRegressor()
tree.fit(X_train,y_train)
print("Decesion Tree")
print(tree.score(X_test,y_test))
# D-Tree hyperparameter tuning.......... finally uncomment
# from sklearn.model_selection import GridSearchCV
# param={"criterion":['squared_error']}
# clasfr=GridSearchCV(tree,param)
# clasfr.fit(X_train,y_train)
# print("D-Tree Hyperparameter Tuning best Score:")
# print(clasfr.best_score_)
