import numpy as np
import openpyxl
import pandas as pd
import scipy
import scipy.spatial.distance as ssd
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn import metrics


from sklearn.metrics import f1_score, precision_score

import pandas as pd
import itertools
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df.rename(columns = {'Churn':'label_1'}, inplace = True)

df['TotalCharges'] = pd.to_numeric(df.TotalCharges, errors='coerce')

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

na_columns = missing_values_table(df, na_name=True)


df.drop(labels=df[df['tenure'] == 0].index, axis=0, inplace=True)
df[df['tenure'] == 0].index

def object_to_int(dataframe_series):
    if dataframe_series.dtype=='object':
        dataframe_series = LabelEncoder().fit_transform(dataframe_series)
    return dataframe_series

df = df.apply(lambda x: object_to_int(x))

print(df['label_1'].value_counts())

df = df.drop(['customerID', 'SeniorCitizen', 'PhoneService', 'Dependents', 'DeviceProtection'], axis = 1)


# df = df.loc[:300]


df['label_1'] = df['label_1'].apply(lambda x: -1 if x == 0 else 1) 
# df['label_2'] = df['label_2'].apply(lambda x: -1 if x == 0 else 1) 


df['label_1'] = df['label_1'].astype(float)
# df['label_2'] = df['label_2'].astype(float)

# label 1
import random
import math

######creating excel files done ################################
def votting(predictions):          #this function use in boosting
    final_pred=[]
    for i in range(predictions.shape[1]):
        lst=list(predictions[:,i])
        majority_vote=max(set(lst), key=lst.count)
        final_pred.append(majority_vote)
    return np.array(final_pred)

number_of_columns=df.shape[1]-1 # -1 bcs class columns isnt considered
number_of_iteration=10 # boosting iter

X_DataFrame=df.drop(columns=['label_1'])
X = X_DataFrame.to_numpy()
y=df["label_1"].to_numpy()

from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split( df, test_size=0.33, random_state=42, stratify=df["label_1"])

positive_class=df_train[df_train["label_1"] == 1]
# positive_class = positive_class.drop(["label_2"], axis=1)

negative_class = df_train[df_train["label_1"] == -1]
# negative_class = negative_class.drop(["label_2"], axis=1)

negative_array=negative_class.drop(columns=['label_1']).to_numpy() #array of negative data without class columns
positive_array=positive_class.drop(columns=['label_1']).to_numpy()

imbalance_ratio = len(negative_class)/len(positive_class)

number_of_positive_data=len(positive_class)
max_cardinal=math.ceil(imbalance_ratio)+1
store=[] # all the clusters that reached to cardinality are stored in this list 

K=len(negative_class) #Initialize #K denote number of clusters
list_of_index_cant_merge=[] # indices that cannot be merged

index_pos=[x for x in range(len(positive_class))]

index_in_each_cluster={}
centers=np.zeros((K,number_of_columns))
for index in range(K):
    index_in_each_cluster[index]=[index]
    centers[index,]=np.mean(negative_array[index_in_each_cluster[index],],axis=0)
index_in_each_cluster=pd.Series(index_in_each_cluster)

print(negative_array.shape)

x_pos_dict={}
for element in itertools.product(range(positive_array.shape[0]),range(positive_array.shape[1])):
    x_pos_dict[element[0],element[1]]=positive_array[element[0]][element[1]]


dim=list(range(number_of_columns))
I=list(range(len(positive_array)))

from pyomo.environ import *

def opt():
    
    model = ConcreteModel()
    model.dual = Suffix(direction=Suffix.IMPORT)

    # Step 1: Define index sets
    J=[x for x in index_in_each_cluster[minimum_dist[0]]+index_in_each_cluster[minimum_dist[1]]]

    # Parameter Definition
    x_neg_dict={}
    for element in itertools.product(index_in_each_cluster[minimum_dist[0]]+index_in_each_cluster[minimum_dist[1]],range(number_of_columns)):
        x_neg_dict[element[0],element[1]]=negative_array[element[0]][element[1]]


    # Step 2: Define the decision 
    model.sigma=Var(within=Reals)
    model.p=Var(J,dim,within=Reals)
    model.q=Var(J,within=Reals)

    # Step 3: Define Objective
    model.obj=Objective(rule=model.sigma,sense=maximize)

    # Step 4: Constraints
    @model.Constraint(J)
    def CNST1(m, j):
        return sum([model.p[j,d]*x_neg_dict[(j,d)] for d in dim]) +model.q[j] <= -model.sigma

    @model.Constraint(I,J)
    def CNST2(m, i,j):
         return  sum([model.p[j,d]*x_pos_dict[(i,d)] for d in dim]) +model.q[j] >=0
    @model.Constraint()
    def CNST3(m):
         return  model.sigma<=1

    SolverFactory('cbc').solve(model)
    star = model.sigma()
    return star

while K>=2:
    

    dist = ssd.squareform(ssd.pdist(centers, 'euclidean'))
    max_element=dist.max()+1
    for element in list_of_index_cant_merge:
        for item in itertools.product(range(K),range(K)):
            if set(element).issubset(set(index_in_each_cluster[item[0]]+index_in_each_cluster[item[1]])):
                dist[item[0],item[1]]=max_element
                dist[item[1],item[0]]=max_element
    dist=np.where(dist!=0, dist, max_element)
    if (np.array_equal(np.ones(dist.shape)*max_element,dist)):
        print("less than "+ str(K)+" negative clusters is not possible")
        break

    minimum_dist=np.unravel_index(dist.argmin(), dist.shape)
    sigma_star = opt()

    ######################clustering start####################

    if (sigma_star==0):
        list_of_index_cant_merge.append(index_in_each_cluster[minimum_dist[0]]+index_in_each_cluster[minimum_dist[1]])
        print("in this step merging isnt occure")

    else:
        K=K-1;
        merge_index=[]
        merge_index=index_in_each_cluster[minimum_dist[0]]+index_in_each_cluster[minimum_dist[1]]
        index_in_each_cluster.drop([minimum_dist[1]],inplace=True)
        index_in_each_cluster[minimum_dist[0]]=merge_index
        if len(merge_index)>=max_cardinal:
            store.append(merge_index)
            index_in_each_cluster.drop([minimum_dist[0]],inplace=True)
            K=K-1
            print("IN THIS STEP max CARDINALITY ACHIEVE",merge_index)
        index_in_each_cluster.reset_index(drop=True,inplace=True)


        centers=np.zeros((K,number_of_columns))

        for index in range(K):
            centers[index,]=np.mean(negative_array[index_in_each_cluster[index],],axis=0)
        if (len(centers)==number_of_positive_data-len(store)):
            print("desired clusters are achieved")
            break
    ######### clustering done! :) while iteration is ended ############
    ########## in this stage data is prepared (choose nearest point to center in each cluster) ######
    ## first index in each cluster dictionary complete


# In[54]:


for i in range(len(store)):
    index_in_each_cluster[K+i]=store[i]
        
centers=np.zeros((len(index_in_each_cluster),number_of_columns))

for index in range(len(positive_array)):
    centers[index,]=np.mean(negative_array[index_in_each_cluster[index],],axis=0)

index_of_negative_train_data=[] # be aware # this list contain the location of data in negative array not hole data
for item in range(len(centers)):
    location=ssd.cdist(centers[item].reshape(1,-1),negative_array[index_in_each_cluster[item],] , metric='euclidean').argmin()
        # location indicates the location of data in item's cluster so for achieving index of nearest data to each cluster center
        #we us below line (index in each cluster dictionary)
        
    index_of_negative_train_data.append(index_in_each_cluster[item][location])
modified_train=pd.concat([negative_class.iloc[index_of_negative_train_data,:], positive_class]) 
# positive_class are added to the ende of negative train data
X_train=modified_train.drop(['label_1'],axis=1)
Y_train=modified_train['label_1']
y_test=df_test['label_1']
x_test=df_test.drop(['label_1'],axis=1)

print(modified_train)


print(modified_train['label_1'].value_counts())


results_column_names = ["tn_test"," fp_test", "fn_test", "tp_test","auc_test", 'test precision -1','test recall -1' ,'test f1-score -1','test support -1','test precision 1','test recall 1','test f1-score 1','test support 1']
results_column_names_with_train = ["tn_test"," fp_test", "fn_test", "tp_test","tn_train"," fp_train", "fn_train", "tp_train","auc_test","auc_train",'train precision -1','train recall -1'
               ,'train f1-score -1','train support -1','train precision 1','train recall 1' ,'train f1-score 1','train support 1', 'test precision -1','test recall -1' ,'test f1-score -1','test support -1','test precision 1','test recall 1','test f1-score 1','test support 1']
# FOR EACH CLASSIFIER CREAT ONE EXCEL FILE WHICH CONATIN RESULTS PER CV ITERATION
result_CART_b=pd.DataFrame(columns = results_column_names)
writer_CART_b = pd.ExcelWriter('max_cardinal_boosting_results_CART.xlsx')
result_rbf_svm_b=pd.DataFrame(columns = results_column_names)
writer_rbf_b = pd.ExcelWriter('max_cardinal_boosting_results_rbf.xlsx')
result_poly_svm_b=pd.DataFrame(columns = results_column_names)
writer_poly_b = pd.ExcelWriter('max_cardinal_boosting_results_poly.xlsx')
result_linear_svm_b=pd.DataFrame(columns = results_column_names)
writer_linear_b = pd.ExcelWriter('max_cardinal_boosting_results_linear.xlsx')
result_1_knn_b=pd.DataFrame(columns = results_column_names)
writer_1_knn_b = pd.ExcelWriter('max_cardinal_boosting_results_1_knn.xlsx')
result_3_knn_b=pd.DataFrame(columns = results_column_names)
writer_3_knn_b = pd.ExcelWriter('max_cardinal_boosting_results_3_knn.xlsx')
result_5_knn_b=pd.DataFrame(columns = results_column_names)
writer_5_knn_b = pd.ExcelWriter('max_cardinal_boosting_results_5_knn.xlsx')
result_LR_b=pd.DataFrame(columns = results_column_names)
writer_LR_b = pd.ExcelWriter('max_cardinal_boosting_results_LR.xlsx')
result_GBM_50_b=pd.DataFrame(columns = results_column_names)
writer_GBM_50_b = pd.ExcelWriter('max_cardinal_boosting_results_GBM_50.xlsx')
result_GBM_100_b=pd.DataFrame(columns = results_column_names)
writer_GBM_100_b = pd.ExcelWriter('max_cardinal_boosting_results_GBM_100.xlsx')
result_GBM_200_b=pd.DataFrame(columns = results_column_names)
writer_GBM_200_b = pd.ExcelWriter('max_cardinal_boosting_results_GBM_200.xlsx')
result_XGB_b=pd.DataFrame(columns = results_column_names)
writer_XGB_b = pd.ExcelWriter('max_cardinal_boosting_results_XGB.xlsx')
result_RNF_b=pd.DataFrame(columns = results_column_names)
writer_RNF_b = pd.ExcelWriter('max_cardinal_boosting_results_RNF.xlsx')

########### nearest point to center scheme excels######
result_CART=pd.DataFrame(columns = results_column_names_with_train)
writer_CART = pd.ExcelWriter('max_cardinal_results_CART.xlsx')
result_rbf_svm=pd.DataFrame(columns = results_column_names_with_train)
writer_rbf = pd.ExcelWriter('max_cardinal_results_rbf.xlsx')
result_poly_svm=pd.DataFrame(columns = results_column_names_with_train)
writer_poly = pd.ExcelWriter('max_cardinal_results_poly.xlsx')
result_linear_svm=pd.DataFrame(columns = results_column_names_with_train)
writer_linear = pd.ExcelWriter('max_cardinal_results_linear.xlsx')
result_1_knn=pd.DataFrame(columns = results_column_names_with_train)
writer_1_knn = pd.ExcelWriter('max_cardinal_results_1_knn.xlsx')
result_3_knn=pd.DataFrame(columns = results_column_names_with_train)
writer_3_knn = pd.ExcelWriter('max_cardinal_results_3_knn.xlsx')
result_5_knn=pd.DataFrame(columns = results_column_names_with_train)
writer_5_knn = pd.ExcelWriter('max_cardinal_results_5_knn.xlsx')
result_LR=pd.DataFrame(columns = results_column_names_with_train)
writer_LR = pd.ExcelWriter('max_cardinal_results_LR.xlsx')
result_GBM_50=pd.DataFrame(columns = results_column_names_with_train)
writer_GBM_50 = pd.ExcelWriter('max_cardinal_results_GBM_50.xlsx')
result_GBM_100=pd.DataFrame(columns = results_column_names_with_train)
writer_GBM_100 = pd.ExcelWriter('max_cardinal_results_GBM_100.xlsx')
result_GBM_200=pd.DataFrame(columns = results_column_names_with_train)
writer_GBM_200 = pd.ExcelWriter('max_cardinal_results_GBM_200.xlsx')

result_XGB=pd.DataFrame(columns = results_column_names_with_train)
writer_XGB = pd.ExcelWriter('max_cardinal_results_XGB.xlsx')

result_RNF=pd.DataFrame(columns = results_column_names_with_train)
writer_RNF = pd.ExcelWriter('max_cardinal_results_RNF.xlsx')


######## CART CLASSIFIER #######
CART=DecisionTreeClassifier(random_state=1)
CART.fit(X_train,Y_train)
y_pred_test_CART=CART.predict(x_test)
y_pred_train_CART=CART.predict(X_train)
test_report_CART=classification_report(y_test, y_pred_test_CART,output_dict=True)
train_report_CART=classification_report(Y_train, y_pred_train_CART,output_dict=True)
auc_test_CART=metrics.roc_auc_score(y_test, y_pred_test_CART)
tn_test_CART, fp_test_CART, fn_test_CART, tp_test_CART = confusion_matrix(y_test, y_pred_test_CART).ravel()
auc_train_CART=metrics.roc_auc_score(Y_train, y_pred_train_CART)
tn_train_CART, fp_train_CART, fn_train_CART, tp_train_CART = confusion_matrix(Y_train, y_pred_train_CART).ravel()
new_cart_row=pd.Series({"tn_test": tn_test_CART," fp_test":fp_test_CART, "fn_test": fn_test_CART, "tp_test":tp_test_CART,"tn_train": tn_train_CART," fp_train":fp_train_CART, "fn_train": fn_train_CART, "tp_train":tp_train_CART, "auc_test":auc_test_CART,"auc_train":auc_train_CART ,'train precision -1':train_report_CART['-1.0']['precision'],'train recall -1':train_report_CART['-1.0']['recall'] ,'train f1-score -1':train_report_CART['-1.0']['f1-score'],'train support -1':train_report_CART['-1.0']['support'] ,'train precision 1':train_report_CART['1.0']['precision'],'train recall 1':train_report_CART['1.0']['recall'] ,'train f1-score 1':train_report_CART['1.0']['f1-score'],'train support 1':train_report_CART['1.0']['support'], 'test precision -1':test_report_CART['-1.0']['precision'],'test recall -1':test_report_CART['-1.0']['recall'] ,'test f1-score -1':test_report_CART['-1.0']['f1-score'],'test support -1':test_report_CART['-1.0']['support'] ,'test precision 1':test_report_CART['1.0']['precision'],'test recall 1':test_report_CART['1.0']['recall'],'test f1-score 1':test_report_CART['1.0']['f1-score'],'test support 1':test_report_CART['1.0']['support']})
result_CART= result_CART.append(new_cart_row,ignore_index=True)
#     ######## RBF SVM ###############
RBF=SVC(probability=True)  # default kernel is Gaussian Radial Basis Function
RBF.fit(X_train,Y_train)
y_pred_test_RBF=RBF.predict(x_test)
y_pred_train_RBF=RBF.predict(X_train)
test_report_RBF=classification_report(y_test, y_pred_test_RBF,output_dict=True)
train_report_RBF=classification_report(Y_train, y_pred_train_RBF,output_dict=True)
auc_test_RBF=metrics.roc_auc_score(y_test, y_pred_test_RBF)
tn_test_RBF, fp_test_RBF, fn_test_RBF, tp_test_RBF = confusion_matrix(y_test, y_pred_test_RBF).ravel()
auc_train_RBF=metrics.roc_auc_score(Y_train, y_pred_train_RBF)
tn_train_RBF, fp_train_RBF, fn_train_RBF, tp_train_RBF = confusion_matrix(Y_train, y_pred_train_RBF).ravel()
new_RBF_row=pd.Series({"tn_test": tn_test_RBF," fp_test":fp_test_RBF, "fn_test": fn_test_RBF, "tp_test":tp_test_RBF,"tn_train": tn_train_RBF," fp_train":fp_train_RBF, "fn_train": fn_train_RBF, "tp_train":tp_train_RBF, "auc_test":auc_test_RBF,"auc_train":auc_train_RBF ,'train precision -1':train_report_RBF['-1.0']['precision'],'train recall -1':train_report_RBF['-1.0']['recall'] ,'train f1-score -1':train_report_RBF['-1.0']['f1-score'],'train support -1':train_report_RBF['-1.0']['support'] ,'train precision 1':train_report_RBF['1.0']['precision'],'train recall 1':train_report_RBF['1.0']['recall'] ,'train f1-score 1':train_report_RBF['1.0']['f1-score'],'train support 1':train_report_RBF['1.0']['support'], 'test precision -1':test_report_RBF['-1.0']['precision'],'test recall -1':test_report_RBF['-1.0']['recall'] ,'test f1-score -1':test_report_RBF['-1.0']['f1-score'],'test support -1':test_report_RBF['-1.0']['support'] ,'test precision 1':test_report_RBF['1.0']['precision'],'test recall 1':test_report_RBF['1.0']['recall'],'test f1-score 1':test_report_RBF['1.0']['f1-score'],'test support 1':test_report_RBF['1.0']['support']})
result_rbf_svm= result_rbf_svm.append(new_RBF_row,ignore_index=True)   
#     ###### LINEAR #########
LINEAR=SVC(kernel='linear', probability=True)
LINEAR.fit(X_train,Y_train)
y_pred_test_LINEAR=LINEAR.predict(x_test)
y_pred_train_LINEAR=LINEAR.predict(X_train)
test_report_LINEAR=classification_report(y_test, y_pred_test_LINEAR,output_dict=True)
train_report_LINEAR=classification_report(Y_train, y_pred_train_LINEAR,output_dict=True)
auc_test_LINEAR=metrics.roc_auc_score(y_test, y_pred_test_LINEAR)
tn_test_LINEAR, fp_test_LINEAR, fn_test_LINEAR, tp_test_LINEAR = confusion_matrix(y_test, y_pred_test_LINEAR).ravel()
auc_train_LINEAR=metrics.roc_auc_score(Y_train, y_pred_train_LINEAR)
tn_train_LINEAR, fp_train_LINEAR, fn_train_LINEAR, tp_train_LINEAR = confusion_matrix(Y_train, y_pred_train_LINEAR).ravel()
new_LINEAR_row=pd.Series({"tn_test": tn_test_LINEAR," fp_test":fp_test_LINEAR, "fn_test": fn_test_LINEAR, "tp_test":tp_test_LINEAR,"tn_train": tn_train_LINEAR," fp_train":fp_train_LINEAR, "fn_train": fn_train_LINEAR, "tp_train":tp_train_LINEAR, "auc_test":auc_test_LINEAR,"auc_train":auc_train_LINEAR ,'train precision -1':train_report_LINEAR['-1.0']['precision'],'train recall -1':train_report_LINEAR['-1.0']['recall'] ,'train f1-score -1':train_report_LINEAR['-1.0']['f1-score'],'train support -1':train_report_LINEAR['-1.0']['support'] ,'train precision 1':train_report_LINEAR['1.0']['precision'],'train recall 1':train_report_LINEAR['1.0']['recall'] ,'train f1-score 1':train_report_LINEAR['1.0']['f1-score'],'train support 1':train_report_LINEAR['1.0']['support'], 'test precision -1':test_report_LINEAR['-1.0']['precision'],'test recall -1':test_report_LINEAR['-1.0']['recall'] ,'test f1-score -1':test_report_LINEAR['-1.0']['f1-score'],'test support -1':test_report_LINEAR['-1.0']['support'] ,'test precision 1':test_report_LINEAR['1.0']['precision'],'test recall 1':test_report_LINEAR['1.0']['recall'],'test f1-score 1':test_report_LINEAR['1.0']['f1-score'],'test support 1':test_report_LINEAR['1.0']['support']})
result_linear_svm= result_linear_svm.append(new_LINEAR_row,ignore_index=True)
#     ###### poly #######
POLY=SVC(kernel='poly')
POLY.fit(X_train,Y_train)
y_pred_test_POLY=POLY.predict(x_test)
y_pred_train_POLY=POLY.predict(X_train)
test_report_POLY=classification_report(y_test, y_pred_test_POLY,output_dict=True)
train_report_POLY=classification_report(Y_train, y_pred_train_POLY,output_dict=True)
auc_test_POLY=roc_auc_score(y_test, y_pred_test_POLY)
tn_test_POLY, fp_test_POLY, fn_test_POLY, tp_test_POLY = confusion_matrix(y_test, y_pred_test_POLY).ravel()
auc_train_POLY=roc_auc_score(Y_train, y_pred_train_POLY)
tn_train_POLY, fp_train_POLY, fn_train_POLY, tp_train_POLY = confusion_matrix(Y_train, y_pred_train_POLY).ravel()
new_POLY_row=pd.Series({"tn_test": tn_test_POLY," fp_test":fp_test_POLY, "fn_test": fn_test_POLY, "tp_test":tp_test_POLY,"tn_train": tn_train_POLY," fp_train":fp_train_POLY, "fn_train": fn_train_POLY, "tp_train":tp_train_POLY, "auc_test":auc_test_POLY,"auc_train":auc_train_POLY ,'train precision -1':train_report_POLY['-1.0']['precision'],'train recall -1':train_report_POLY['-1.0']['recall'] ,'train f1-score -1':train_report_POLY['-1.0']['f1-score'],'train support -1':train_report_POLY['-1.0']['support'] ,'train precision 1':train_report_POLY['1.0']['precision'],'train recall 1':train_report_POLY['1.0']['recall'] ,'train f1-score 1':train_report_POLY['1.0']['f1-score'],'train support 1':train_report_POLY['1.0']['support'], 'test precision -1':test_report_POLY['-1.0']['precision'],'test recall -1':test_report_POLY['-1.0']['recall'] ,'test f1-score -1':test_report_POLY['-1.0']['f1-score'],'test support -1':test_report_POLY['-1.0']['support'] ,'test precision 1':test_report_POLY['1.0']['precision'],'test recall 1':test_report_POLY['1.0']['recall'],'test f1-score 1':test_report_POLY['1.0']['f1-score'],'test support 1':test_report_POLY['1.0']['support']})
result_poly_svm= result_poly_svm.append(new_POLY_row,ignore_index=True)
###### 1 KNN ##########
KNN_1=KNeighborsClassifier(n_neighbors=1)
KNN_1.fit(X_train,Y_train)
y_pred_test_1_KNN=KNN_1.predict(x_test)
y_pred_train_1_KNN=KNN_1.predict(X_train)
test_report_1_KNN=classification_report(y_test, y_pred_test_1_KNN,output_dict=True)
train_report_1_KNN=classification_report(Y_train, y_pred_train_1_KNN,output_dict=True)
auc_test_1_KNN=metrics.roc_auc_score(y_test, y_pred_test_1_KNN)
tn_test_1_KNN, fp_test_1_KNN, fn_test_1_KNN, tp_test_1_KNN = confusion_matrix(y_test, y_pred_test_1_KNN).ravel()
auc_train_1_KNN=metrics.roc_auc_score(Y_train, y_pred_train_1_KNN)
tn_train_1_KNN, fp_train_1_KNN, fn_train_1_KNN, tp_train_1_KNN= confusion_matrix(Y_train, y_pred_train_1_KNN).ravel()
new_1_KNN_row=pd.Series({"tn_test": tn_test_1_KNN," fp_test":fp_test_1_KNN, "fn_test": fn_test_1_KNN, "tp_test":tp_test_1_KNN,"tn_train": tn_train_1_KNN," fp_train":fp_train_1_KNN, "fn_train": fn_train_1_KNN, "tp_train":tp_train_1_KNN, "auc_test":auc_test_1_KNN,"auc_train":auc_train_1_KNN ,'train precision -1':train_report_1_KNN['-1.0']['precision'],'train recall -1':train_report_1_KNN['-1.0']['recall'] ,'train f1-score -1':train_report_1_KNN['-1.0']['f1-score'],'train support -1':train_report_1_KNN['-1.0']['support'] ,'train precision 1':train_report_1_KNN['1.0']['precision'],'train recall 1':train_report_1_KNN['1.0']['recall'] ,'train f1-score 1':train_report_1_KNN['1.0']['f1-score'],'train support 1':train_report_1_KNN['1.0']['support'], 'test precision -1':test_report_1_KNN['-1.0']['precision'],'test recall -1':test_report_1_KNN['-1.0']['recall'] ,'test f1-score -1':test_report_1_KNN['-1.0']['f1-score'],'test support -1':test_report_1_KNN['-1.0']['support'] ,'test precision 1':test_report_1_KNN['1.0']['precision'],'test recall 1':test_report_1_KNN['1.0']['recall'],'test f1-score 1':test_report_1_KNN['1.0']['f1-score'],'test support 1':test_report_1_KNN['1.0']['support']})
result_1_knn= result_1_knn.append(new_1_KNN_row,ignore_index=True)    
#### 3 KNN ######
KNN_3=KNeighborsClassifier(n_neighbors=3)
KNN_3.fit(X_train,Y_train)
y_pred_test_3_KNN=KNN_3.predict(x_test)
y_pred_train_3_KNN=KNN_3.predict(X_train)
test_report_3_KNN=classification_report(y_test, y_pred_test_3_KNN,output_dict=True)
train_report_3_KNN=classification_report(Y_train, y_pred_train_3_KNN,output_dict=True)
auc_test_3_KNN=metrics.roc_auc_score(y_test, y_pred_test_3_KNN)
tn_test_3_KNN, fp_test_3_KNN, fn_test_3_KNN, tp_test_3_KNN = confusion_matrix(y_test, y_pred_test_3_KNN).ravel()
auc_train_3_KNN=metrics.roc_auc_score(Y_train, y_pred_train_3_KNN)
tn_train_3_KNN, fp_train_3_KNN, fn_train_3_KNN, tp_train_3_KNN= confusion_matrix(Y_train, y_pred_train_3_KNN).ravel()
new_3_KNN_row=pd.Series({"tn_test": tn_test_3_KNN," fp_test":fp_test_3_KNN, "fn_test": fn_test_3_KNN, "tp_test":tp_test_3_KNN,"tn_train": tn_train_3_KNN," fp_train":fp_train_3_KNN, "fn_train": fn_train_3_KNN, "tp_train":tp_train_3_KNN, "auc_test":auc_test_3_KNN,"auc_train":auc_train_3_KNN ,'train precision -1':train_report_3_KNN['-1.0']['precision'],'train recall -1':train_report_3_KNN['-1.0']['recall'] ,'train f1-score -1':train_report_3_KNN['-1.0']['f1-score'],'train support -1':train_report_3_KNN['-1.0']['support'] ,'train precision 1':train_report_3_KNN['1.0']['precision'],'train recall 1':train_report_3_KNN['1.0']['recall'] ,'train f1-score 1':train_report_3_KNN['1.0']['f1-score'],'train support 1':train_report_3_KNN['1.0']['support'], 'test precision -1':test_report_3_KNN['-1.0']['precision'],'test recall -1':test_report_3_KNN['-1.0']['recall'] ,'test f1-score -1':test_report_3_KNN['-1.0']['f1-score'],'test support -1':test_report_3_KNN['-1.0']['support'] ,'test precision 1':test_report_3_KNN['1.0']['precision'],'test recall 1':test_report_3_KNN['1.0']['recall'],'test f1-score 1':test_report_3_KNN['1.0']['f1-score'],'test support 1':test_report_3_KNN['1.0']['support']})
result_3_knn= result_3_knn.append(new_3_KNN_row,ignore_index=True)    
##### 5 KNN #######
KNN_5=KNeighborsClassifier(n_neighbors=5)
KNN_5.fit(X_train,Y_train)
y_pred_test_5_KNN=KNN_5.predict(x_test)
y_pred_train_5_KNN=KNN_5.predict(X_train)
test_report_5_KNN=classification_report(y_test, y_pred_test_5_KNN,output_dict=True)
train_report_5_KNN=classification_report(Y_train, y_pred_train_5_KNN,output_dict=True)
auc_test_5_KNN=metrics.roc_auc_score(y_test, y_pred_test_5_KNN)
tn_test_5_KNN, fp_test_5_KNN, fn_test_5_KNN, tp_test_5_KNN = confusion_matrix(y_test, y_pred_test_5_KNN).ravel()
auc_train_5_KNN=metrics.roc_auc_score(Y_train, y_pred_train_5_KNN)
tn_train_5_KNN, fp_train_5_KNN, fn_train_5_KNN, tp_train_5_KNN= confusion_matrix(Y_train, y_pred_train_5_KNN).ravel()
new_5_KNN_row=pd.Series({"tn_test": tn_test_5_KNN," fp_test":fp_test_5_KNN, "fn_test": fn_test_5_KNN, "tp_test":tp_test_5_KNN,"tn_train": tn_train_5_KNN," fp_train":fp_train_5_KNN, "fn_train": fn_train_5_KNN, "tp_train":tp_train_5_KNN, "auc_test":auc_test_5_KNN,"auc_train":auc_train_5_KNN ,'train precision -1':train_report_5_KNN['-1.0']['precision'],'train recall -1':train_report_5_KNN['-1.0']['recall'] ,'train f1-score -1':train_report_5_KNN['-1.0']['f1-score'],'train support -1':train_report_5_KNN['-1.0']['support'] ,'train precision 1':train_report_5_KNN['1.0']['precision'],'train recall 1':train_report_5_KNN['1.0']['recall'] ,'train f1-score 1':train_report_5_KNN['1.0']['f1-score'],'train support 1':train_report_5_KNN['1.0']['support'], 'test precision -1':test_report_5_KNN['-1.0']['precision'],'test recall -1':test_report_5_KNN['-1.0']['recall'] ,'test f1-score -1':test_report_5_KNN['-1.0']['f1-score'],'test support -1':test_report_5_KNN['-1.0']['support'] ,'test precision 1':test_report_5_KNN['1.0']['precision'],'test recall 1':test_report_5_KNN['1.0']['recall'],'test f1-score 1':test_report_5_KNN['1.0']['f1-score'],'test support 1':test_report_5_KNN['1.0']['support']})
result_5_knn= result_5_knn.append(new_5_KNN_row,ignore_index=True)    
#     ##### GBM 50 #######

GBM_50=GradientBoostingClassifier(n_estimators=50, random_state=1)
GBM_50.fit(X_train,Y_train)
y_pred_test_GBM_50=GBM_50.predict(x_test)
y_pred_train_GBM_50=GBM_50.predict(X_train)
test_report_GBM_50=classification_report(y_test, y_pred_test_GBM_50,output_dict=True)
train_report_GBM_50=classification_report(Y_train, y_pred_train_GBM_50,output_dict=True)
auc_test_GBM_50=metrics.roc_auc_score(y_test, y_pred_test_GBM_50)
tn_test_GBM_50, fp_test_GBM_50, fn_test_GBM_50, tp_test_GBM_50 = confusion_matrix(y_test, y_pred_test_GBM_50).ravel()
auc_train_GBM_50=metrics.roc_auc_score(Y_train, y_pred_train_GBM_50)
tn_train_GBM_50, fp_train_GBM_50, fn_train_GBM_50, tp_train_GBM_50= confusion_matrix(Y_train, y_pred_train_GBM_50).ravel()
new_GBM_50_row=pd.Series({"tn_test": tn_test_GBM_50," fp_test":fp_test_GBM_50, "fn_test": fn_test_GBM_50, "tp_test":tp_test_GBM_50,"tn_train": tn_train_GBM_50," fp_train":fp_train_GBM_50, "fn_train": fn_train_GBM_50, "tp_train":tp_train_GBM_50, "auc_test":auc_test_GBM_50,"auc_train":auc_train_GBM_50 ,'train precision -1':train_report_GBM_50['-1.0']['precision'],'train recall -1':train_report_GBM_50['-1.0']['recall'] ,'train f1-score -1':train_report_GBM_50['-1.0']['f1-score'],'train support -1':train_report_GBM_50['-1.0']['support'] ,'train precision 1':train_report_GBM_50['1.0']['precision'],'train recall 1':train_report_GBM_50['1.0']['recall'] ,'train f1-score 1':train_report_GBM_50['1.0']['f1-score'],'train support 1':train_report_GBM_50['1.0']['support'], 'test precision -1':test_report_GBM_50['-1.0']['precision'],'test recall -1':test_report_GBM_50['-1.0']['recall'] ,'test f1-score -1':test_report_GBM_50['-1.0']['f1-score'],'test support -1':test_report_GBM_50['-1.0']['support'] ,'test precision 1':test_report_GBM_50['1.0']['precision'],'test recall 1':test_report_GBM_50['1.0']['recall'],'test f1-score 1':test_report_GBM_50['1.0']['f1-score'],'test support 1':test_report_GBM_50['1.0']['support']})
result_GBM_50= result_GBM_50.append(new_GBM_50_row,ignore_index=True)  
#     ########## GBM 100 ######

GBM_100=GradientBoostingClassifier(n_estimators=100, random_state=1)
GBM_100.fit(X_train,Y_train)
y_pred_test_GBM_100=GBM_100.predict(x_test)
y_pred_train_GBM_100=GBM_100.predict(X_train)
test_report_GBM_100=classification_report(y_test, y_pred_test_GBM_100,output_dict=True)
train_report_GBM_100=classification_report(Y_train, y_pred_train_GBM_100,output_dict=True)
auc_test_GBM_100=metrics.roc_auc_score(y_test, y_pred_test_GBM_100)
tn_test_GBM_100, fp_test_GBM_100, fn_test_GBM_100, tp_test_GBM_100 = confusion_matrix(y_test, y_pred_test_GBM_100).ravel()
auc_train_GBM_100=metrics.roc_auc_score(Y_train, y_pred_train_GBM_100)
tn_train_GBM_100, fp_train_GBM_100, fn_train_GBM_100, tp_train_GBM_100= confusion_matrix(Y_train, y_pred_train_GBM_100).ravel()
new_GBM_100_row=pd.Series({"tn_test": tn_test_GBM_100," fp_test":fp_test_GBM_100, "fn_test": fn_test_GBM_100, "tp_test":tp_test_GBM_100,"tn_train": tn_train_GBM_100," fp_train":fp_train_GBM_100, "fn_train": fn_train_GBM_100, "tp_train":tp_train_GBM_100, "auc_test":auc_test_GBM_100,"auc_train":auc_train_GBM_100 ,'train precision -1':train_report_GBM_100['-1.0']['precision'],'train recall -1':train_report_GBM_100['-1.0']['recall'] ,'train f1-score -1':train_report_GBM_100['-1.0']['f1-score'],'train support -1':train_report_GBM_100['-1.0']['support'] ,'train precision 1':train_report_GBM_100['1.0']['precision'],'train recall 1':train_report_GBM_100['1.0']['recall'] ,'train f1-score 1':train_report_GBM_100['1.0']['f1-score'],'train support 1':train_report_GBM_100['1.0']['support'], 'test precision -1':test_report_GBM_100['-1.0']['precision'],'test recall -1':test_report_GBM_100['-1.0']['recall'] ,'test f1-score -1':test_report_GBM_100['-1.0']['f1-score'],'test support -1':test_report_GBM_100['-1.0']['support'] ,'test precision 1':test_report_GBM_100['1.0']['precision'],'test recall 1':test_report_GBM_100['1.0']['recall'],'test f1-score 1':test_report_GBM_100['1.0']['f1-score'],'test support 1':test_report_GBM_100['1.0']['support']})
result_GBM_100= result_GBM_100.append(new_GBM_100_row,ignore_index=True)  
#     ########## GBM 200 ######

GBM_200=GradientBoostingClassifier(n_estimators=200, random_state=1)
GBM_200.fit(X_train,Y_train)
y_pred_test_GBM_200=GBM_200.predict(x_test)
y_pred_train_GBM_200=GBM_200.predict(X_train)
test_report_GBM_200=classification_report(y_test, y_pred_test_GBM_200,output_dict=True)
train_report_GBM_200=classification_report(Y_train, y_pred_train_GBM_200,output_dict=True)
auc_test_GBM_200=metrics.roc_auc_score(y_test, y_pred_test_GBM_200)
tn_test_GBM_200, fp_test_GBM_200, fn_test_GBM_200, tp_test_GBM_200 = confusion_matrix(y_test, y_pred_test_GBM_200).ravel()
auc_train_GBM_200=metrics.roc_auc_score(Y_train, y_pred_train_GBM_200)
tn_train_GBM_200, fp_train_GBM_200, fn_train_GBM_200, tp_train_GBM_200= confusion_matrix(Y_train, y_pred_train_GBM_200).ravel()
new_GBM_200_row=pd.Series({"tn_test": tn_test_GBM_200," fp_test":fp_test_GBM_200, "fn_test": fn_test_GBM_200, "tp_test":tp_test_GBM_200,"tn_train": tn_train_GBM_200," fp_train":fp_train_GBM_200, "fn_train": fn_train_GBM_200, "tp_train":tp_train_GBM_200, "auc_test":auc_test_GBM_200,"auc_train":auc_train_GBM_200 ,'train precision -1':train_report_GBM_200['-1.0']['precision'],'train recall -1':train_report_GBM_200['-1.0']['recall'] ,'train f1-score -1':train_report_GBM_200['-1.0']['f1-score'],'train support -1':train_report_GBM_200['-1.0']['support'] ,'train precision 1':train_report_GBM_200['1.0']['precision'],'train recall 1':train_report_GBM_200['1.0']['recall'] ,'train f1-score 1':train_report_GBM_200['1.0']['f1-score'],'train support 1':train_report_GBM_200['1.0']['support'], 'test precision -1':test_report_GBM_200['-1.0']['precision'],'test recall -1':test_report_GBM_200['-1.0']['recall'] ,'test f1-score -1':test_report_GBM_200['-1.0']['f1-score'],'test support -1':test_report_GBM_200['-1.0']['support'] ,'test precision 1':test_report_GBM_200['1.0']['precision'],'test recall 1':test_report_GBM_200['1.0']['recall'],'test f1-score 1':test_report_GBM_200['1.0']['f1-score'],'test support 1':test_report_GBM_200['1.0']['support']})
result_GBM_200= result_GBM_200.append(new_GBM_200_row,ignore_index=True)  
# ######### LR #######    
LR=LogisticRegression(random_state=1)
LR.fit(X_train,Y_train)
y_pred_test_LR=LR.predict(x_test)
y_pred_train_LR=LR.predict(X_train)
test_report_LR=classification_report(y_test, y_pred_test_LR,output_dict=True)
train_report_LR=classification_report(Y_train, y_pred_train_LR,output_dict=True)
auc_test_LR=metrics.roc_auc_score(y_test, y_pred_test_LR)
tn_test_LR, fp_test_LR, fn_test_LR, tp_test_LR= confusion_matrix(y_test, y_pred_test_LR).ravel()
auc_train_LR=metrics.roc_auc_score(Y_train, y_pred_train_LR)
tn_train_LR, fp_train_LR, fn_train_LR, tp_train_LR= confusion_matrix(Y_train, y_pred_train_LR).ravel()
new_LR_row=pd.Series({"tn_test": tn_test_LR," fp_test":fp_test_LR, "fn_test": fn_test_LR, "tp_test":tp_test_LR,"tn_train": tn_train_LR," fp_train":fp_train_LR, "fn_train": fn_train_LR, "tp_train":tp_train_LR, "auc_test":auc_test_LR,"auc_train":auc_train_LR ,'train precision -1':train_report_LR['-1.0']['precision'],'train recall -1':train_report_LR['-1.0']['recall'] ,'train f1-score -1':train_report_LR['-1.0']['f1-score'],'train support -1':train_report_LR['-1.0']['support'] ,'train precision 1':train_report_LR['1.0']['precision'],'train recall 1':train_report_LR['1.0']['recall'] ,'train f1-score 1':train_report_LR['1.0']['f1-score'],'train support 1':train_report_LR['1.0']['support'], 'test precision -1':test_report_LR['-1.0']['precision'],'test recall -1':test_report_LR['-1.0']['recall'] ,'test f1-score -1':test_report_LR['-1.0']['f1-score'],'test support -1':test_report_LR['-1.0']['support'] ,'test precision 1':test_report_LR['1.0']['precision'],'test recall 1':test_report_LR['1.0']['recall'],'test f1-score 1':test_report_LR['1.0']['f1-score'],'test support 1':test_report_LR['1.0']['support']})
result_LR= result_LR.append(new_LR_row,ignore_index=True) 

########## XGB ##################
xgb_model=XGBClassifier(random_state=600)
xgb_model.fit(X_train,Y_train)
y_pred_test_XGB=xgb_model.predict(x_test)
y_pred_train_XGB=xgb_model.predict(X_train)
test_report_XGB=classification_report(y_test, y_pred_test_XGB,output_dict=True)
train_report_XGB=classification_report(Y_train, y_pred_train_XGB,output_dict=True)
auc_test_XGB=metrics.roc_auc_score(y_test, y_pred_test_XGB)
tn_test_XGB, fp_test_XGB, fn_test_XGB, tp_test_XGB= confusion_matrix(y_test, y_pred_test_XGB).ravel()
auc_train_XGB=metrics.roc_auc_score(Y_train, y_pred_train_XGB)
tn_train_XGB, fp_train_XGB, fn_train_XGB, tp_train_XGB= confusion_matrix(Y_train, y_pred_train_XGB).ravel()
new_XGB_row=pd.Series({"tn_test": tn_test_XGB," fp_test":fp_test_XGB, "fn_test": fn_test_XGB, "tp_test":tp_test_XGB,"tn_train": tn_train_XGB," fp_train":fp_train_XGB, "fn_train": fn_train_XGB, "tp_train":tp_train_XGB, "auc_test":auc_test_XGB,"auc_train":auc_train_XGB ,'train precision -1':train_report_XGB['-1.0']['precision'],'train recall -1':train_report_XGB['-1.0']['recall'] ,'train f1-score -1':train_report_XGB['-1.0']['f1-score'],'train support -1':train_report_XGB['-1.0']['support'] ,'train precision 1':train_report_XGB['1.0']['precision'],'train recall 1':train_report_XGB['1.0']['recall'] ,'train f1-score 1':train_report_XGB['1.0']['f1-score'],'train support 1':train_report_XGB['1.0']['support'], 'test precision -1':test_report_XGB['-1.0']['precision'],'test recall -1':test_report_XGB['-1.0']['recall'] ,'test f1-score -1':test_report_XGB['-1.0']['f1-score'],'test support -1':test_report_XGB['-1.0']['support'] ,'test precision 1':test_report_XGB['1.0']['precision'],'test recall 1':test_report_XGB['1.0']['recall'],'test f1-score 1':test_report_XGB['1.0']['f1-score'],'test support 1':test_report_XGB['1.0']['support']})
result_XGB= result_XGB.append(new_XGB_row,ignore_index=True) 

############### RNF ###################
RNF = RandomForestClassifier(random_state=700)
RNF.fit(X_train,Y_train)
y_pred_test_RNF=RNF.predict(x_test)
y_pred_train_RNF=RNF.predict(X_train)
test_report_RNF=classification_report(y_test, y_pred_test_RNF,output_dict=True)
train_report_RNF=classification_report(Y_train, y_pred_train_RNF,output_dict=True)
auc_test_RNF=metrics.roc_auc_score(y_test, y_pred_test_RNF)
tn_test_RNF, fp_test_RNF, fn_test_RNF, tp_test_RNF= confusion_matrix(y_test, y_pred_test_RNF).ravel()
auc_train_RNF=metrics.roc_auc_score(Y_train, y_pred_train_RNF)
tn_train_RNF, fp_train_RNF, fn_train_RNF, tp_train_RNF= confusion_matrix(Y_train, y_pred_train_RNF).ravel()
new_RNF_row=pd.Series({"tn_test": tn_test_RNF," fp_test":fp_test_RNF, "fn_test": fn_test_RNF, "tp_test":tp_test_RNF,"tn_train": tn_train_RNF," fp_train":fp_train_RNF, "fn_train": fn_train_RNF, "tp_train":tp_train_RNF, "auc_test":auc_test_RNF,"auc_train":auc_train_RNF ,'train precision -1':train_report_RNF['-1.0']['precision'],'train recall -1':train_report_RNF['-1.0']['recall'] ,'train f1-score -1':train_report_RNF['-1.0']['f1-score'],'train support -1':train_report_RNF['-1.0']['support'] ,'train precision 1':train_report_RNF['1.0']['precision'],'train recall 1':train_report_RNF['1.0']['recall'] ,'train f1-score 1':train_report_RNF['1.0']['f1-score'],'train support 1':train_report_RNF['1.0']['support'], 'test precision -1':test_report_RNF['-1.0']['precision'],'test recall -1':test_report_RNF['-1.0']['recall'] ,'test f1-score -1':test_report_RNF['-1.0']['f1-score'],'test support -1':test_report_RNF['-1.0']['support'] ,'test precision 1':test_report_RNF['1.0']['precision'],'test recall 1':test_report_RNF['1.0']['recall'],'test f1-score 1':test_report_RNF['1.0']['f1-score'],'test support 1':test_report_XGB['1.0']['support']})
result_RNF= result_RNF.append(new_RNF_row,ignore_index=True) 

############# catboost #############
CatBoost = CatBoostClassifier(random_state=700)
CatBoost.fit(X_train,Y_train)
y_pred_test_CatBoost=CatBoost.predict(x_test)
y_pred_train_CatBoost=CatBoost.predict(X_train)
test_report_CatBoost=classification_report(y_test, y_pred_test_CatBoost,output_dict=True)
train_report_CatBoost=classification_report(Y_train, y_pred_train_CatBoost,output_dict=True)
auc_test_CatBoost=metrics.roc_auc_score(y_test, y_pred_test_CatBoost)
tn_test_CatBoost, fp_test_CatBoost, fn_test_CatBoost, tp_test_CatBoost= confusion_matrix(y_test, y_pred_test_CatBoost).ravel()
auc_train_CatBoost=metrics.roc_auc_score(Y_train, y_pred_train_CatBoost)
tn_train_CatBoost, fp_train_CatBoost, fn_train_CatBoost, tp_train_CatBoost= confusion_matrix(Y_train, y_pred_train_CatBoost).ravel()
new_CatBoost_row=pd.Series({"tn_test": tn_test_CatBoost," fp_test":fp_test_CatBoost, "fn_test": fn_test_CatBoost, "tp_test":tp_test_CatBoost,"tn_train": tn_train_CatBoost," fp_train":fp_train_CatBoost, "fn_train": fn_train_CatBoost, "tp_train":tp_train_CatBoost, "auc_test":auc_test_CatBoost,"auc_train":auc_train_CatBoost ,'train precision -1':train_report_CatBoost['-1.0']['precision'],'train recall -1':train_report_CatBoost['-1.0']['recall'] ,'train f1-score -1':train_report_CatBoost['-1.0']['f1-score'],'train support -1':train_report_CatBoost['-1.0']['support'] ,'train precision 1':train_report_CatBoost['1.0']['precision'],'train recall 1':train_report_CatBoost['1.0']['recall'] ,'train f1-score 1':train_report_CatBoost['1.0']['f1-score'],'train support 1':train_report_CatBoost['1.0']['support'], 'test precision -1':test_report_CatBoost['-1.0']['precision'],'test recall -1':test_report_CatBoost['-1.0']['recall'] ,'test f1-score -1':test_report_CatBoost['-1.0']['f1-score'],'test support -1':test_report_CatBoost['-1.0']['support'] ,'test precision 1':test_report_CatBoost['1.0']['precision'],'test recall 1':test_report_CatBoost['1.0']['recall'],'test f1-score 1':test_report_CatBoost['1.0']['f1-score'],'test support 1':test_report_CatBoost['1.0']['support']})

#### BOOSTING SCHEME IS STARTED ######
# BE aware X_test and y_test which define in previous stage, apply in following
########## in this stage 10 times randomly select one data from each cluster and then boosting learning apply ######
prediction_array_CART=np.zeros((number_of_iteration,len(df_test))) #  each row is a prediction of test set 
prediction_array_rbf=np.zeros((number_of_iteration,len(df_test)))
# prediction_array_poly=np.zeros((number_of_iteration,len(df_test)))
prediction_array_linear=np.zeros((number_of_iteration,len(df_test)))
prediction_array_KNN_1=np.zeros((number_of_iteration,len(df_test)))
prediction_array_KNN_3=np.zeros((number_of_iteration,len(df_test)))
prediction_array_KNN_5=np.zeros((number_of_iteration,len(df_test)))
prediction_array_GBM50=np.zeros((number_of_iteration,len(df_test)))
prediction_array_GBM100=np.zeros((number_of_iteration,len(df_test)))
prediction_array_GBM200=np.zeros((number_of_iteration,len(df_test)))
prediction_array_LR=np.zeros((number_of_iteration,len(df_test)))
prediction_array_xgb=np.zeros((number_of_iteration,len(df_test)))
prediction_array_RNF=np.zeros((number_of_iteration,len(df_test)))

###### ALL array of predictions created (these arrays store predictions) #####

for iteration in range(number_of_iteration):
    index_of_negative_train_data=[]
    for i in range(len(index_in_each_cluster)):
        rand=random.randint(0, len(index_in_each_cluster[i])-1)
        index_of_negative_train_data.append(index_in_each_cluster[i][rand])
    modified_train=[]
    modified_train=pd.concat([negative_class.iloc[index_of_negative_train_data,:], positive_class])
    X_train=modified_train.drop(['label_1'],axis=1)
    Y_train=modified_train['label_1']  
    ###### cart classifier ####
    CART=DecisionTreeClassifier(random_state=1) # in previous stage CART in defined but for sure defined again
    CART.fit(X_train,Y_train)
    prediction_array_CART[iteration,]=CART.predict(x_test)
    ######## rbf svm ####
    RBF=SVC(probability=True)
    RBF.fit(X_train,Y_train)
    prediction_array_rbf[iteration,]=RBF.predict(x_test)
    ######## Linear ####
    LINEAR=SVC(kernel='linear', probability=True)
    LINEAR.fit(X_train,Y_train)
    prediction_array_linear[iteration,]=LINEAR.predict(x_test)
    ####### poly #######
#     POLY=SVC(kernel='poly')
#     POLY.fit(X_train,Y_train)
#     prediction_array_poly[iteration,]=POLY.predict(x_test)
    ####### 1knn######
    KNN_1=KNeighborsClassifier(n_neighbors=1)
    KNN_1.fit(X_train,Y_train)
    prediction_array_KNN_1[iteration,]=KNN_1.predict(x_test)
    ##### 3knn ########
    KNN_3=KNeighborsClassifier(n_neighbors=3)
    KNN_3.fit(X_train,Y_train)
    prediction_array_KNN_3[iteration,]=KNN_3.predict(x_test)
    ####### 5 knn ########
    KNN_5=KNeighborsClassifier(n_neighbors=5)
    KNN_5.fit(X_train,Y_train)
    prediction_array_KNN_5[iteration,]=KNN_5.predict(x_test)
    ######### GBM 50 ######
    GBM_50=GradientBoostingClassifier(n_estimators=50, random_state=1)
    GBM_50.fit(X_train,Y_train)
    prediction_array_GBM50[iteration,]=GBM_50.predict(x_test)
    ######### GBM 100 ###
    GBM_100=GradientBoostingClassifier(n_estimators=100, random_state=1)
    GBM_100.fit(X_train,Y_train)
    prediction_array_GBM100[iteration,]=GBM_100.predict(x_test)
    ####GBM 200 ##
    GBM_200=GradientBoostingClassifier(n_estimators=200, random_state=1)
    GBM_200.fit(X_train,Y_train)
    prediction_array_GBM200[iteration,]=GBM_200.predict(x_test)
    ####### LR ######
    LR=LogisticRegression(random_state=1,max_iter=500)
    LR.fit(X_train,Y_train)
    prediction_array_LR[iteration,]=LR.predict(x_test)
    ###### XGBoost #######
    xgb_model = XGBClassifier(random_state=600)
    xgb_model.fit(X_train,Y_train)
    prediction_array_xgb[iteration,] = xgb_model.predict(x_test)
    ###### RNF ###############
    RNF = RandomForestClassifier(random_state=700)
    RNF.fit(X_train,Y_train)
    prediction_array_RNF[iteration,] = RNF.predict(x_test)
    
# # now all arrays of predictions get values so votting start
y_pred_test_CART=votting(prediction_array_CART)
y_pred_test_RBF=votting(prediction_array_rbf)
y_pred_test_LINEAR=votting(prediction_array_linear)
# y_pred_test_POLY=votting(prediction_array_poly)
y_pred_test_3_KNN=votting(prediction_array_KNN_3)
y_pred_test_5_KNN=votting(prediction_array_KNN_5)
y_pred_test_1_KNN=votting(prediction_array_KNN_1)
y_pred_test_GBM_50=votting(prediction_array_GBM50)
y_pred_test_GBM_100=votting(prediction_array_GBM100)
y_pred_test_GBM_200=votting(prediction_array_GBM200)
y_pred_test_LR=votting(prediction_array_LR)
y_pred_test_XGB=votting(prediction_array_xgb)
y_pred_test_RNF=votting(prediction_array_RNF)

# ##### NOW calculate evaluation metrics like previous works #####
test_report_CART=classification_report(y_test, y_pred_test_CART,output_dict=True)


auc_test_CART=metrics.roc_auc_score(y_test, y_pred_test_CART)
tn_test_CART, fp_test_CART, fn_test_CART, tp_test_CART = confusion_matrix(y_test, y_pred_test_CART).ravel()
new_cart_row_b=pd.Series({"tn_test": tn_test_CART," fp_test":fp_test_CART, "fn_test": fn_test_CART,"tp_test":tp_test_CART, "auc_test":auc_test_CART, 'test precision -1':test_report_CART['-1.0']['precision'],'test recall -1':test_report_CART['-1.0']['recall'] ,'test f1-score -1':test_report_CART['-1.0']['f1-score'],'test support -1':test_report_CART['-1.0']['support'] ,'test precision 1':test_report_CART['1.0']['precision'],'test recall 1':test_report_CART['1.0']['recall'],'test f1-score 1':test_report_CART['1.0']['f1-score'],'test support 1':test_report_CART['1.0']['support']})
result_CART_b= result_CART_b.append(new_cart_row_b,ignore_index=True)
# ##
test_report_RBF=classification_report(y_test, y_pred_test_RBF,output_dict=True)
auc_test_RBF=roc_auc_score(y_test, y_pred_test_RBF)
tn_test_RBF, fp_test_RBF, fn_test_RBF, tp_test_RBF = confusion_matrix(y_test, y_pred_test_RBF).ravel()
new_RBF_row_b=pd.Series({"tn_test": tn_test_RBF," fp_test":fp_test_RBF, "fn_test": fn_test_RBF, "tp_test":tp_test_RBF, "auc_test":auc_test_RBF, 'test precision -1':test_report_RBF['-1.0']['precision'],'test recall -1':test_report_RBF['-1.0']['recall'] ,'test f1-score -1':test_report_RBF['-1.0']['f1-score'],'test support -1':test_report_RBF['-1.0']['support'] ,'test precision 1':test_report_RBF['1.0']['precision'],'test recall 1':test_report_RBF['1.0']['recall'],'test f1-score 1':test_report_RBF['1.0']['f1-score'],'test support 1':test_report_RBF['1.0']['support']})
result_rbf_svm_b= result_rbf_svm_b.append(new_RBF_row_b,ignore_index=True)
###
test_report_POLY=classification_report(y_test, y_pred_test_POLY,output_dict=True)
auc_test_POLY=roc_auc_score(y_test, y_pred_test_POLY)
tn_test_POLY, fp_test_POLY, fn_test_POLY, tp_test_POLY = confusion_matrix(y_test, y_pred_test_POLY).ravel()
new_POLY_row_b=pd.Series({"tn_test": tn_test_POLY," fp_test":fp_test_POLY, "fn_test": fn_test_POLY, "tp_test":tp_test_POLY, "auc_test":auc_test_POLY, 'test precision -1':test_report_POLY['-1.0']['precision'],'test recall -1':test_report_POLY['-1.0']['recall'] ,'test f1-score -1':test_report_POLY['-1.0']['f1-score'],'test support -1':test_report_POLY['-1.0']['support'] ,'test precision 1':test_report_POLY['1.0']['precision'],'test recall 1':test_report_POLY['1.0']['recall'],'test f1-score 1':test_report_POLY['1.0']['f1-score'],'test support 1':test_report_POLY['1.0']['support']})
result_poly_svm_b= result_poly_svm_b.append(new_POLY_row_b,ignore_index=True)


# # *CART*
y_pred_proba_CART = CART.predict_proba(x_test)[:][:,1]
df_actual_predicted_CART = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_CART, columns=['y_pred_proba_CART'])], axis=1)
df_actual_predicted_CART.index = y_test.index
fpr, tpr, tr = metrics.roc_curve(df_actual_predicted_CART['label_1'], df_actual_predicted_CART['y_pred_proba_CART'])
auc_CART = metrics.roc_auc_score(df_actual_predicted_CART['label_1'], df_actual_predicted_CART['y_pred_proba_CART'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('DecisionTreeClassifier', size = 10)
# plt.legend()


# # *KNN 1*

y_pred_proba_KNN_1 = KNN_1.predict_proba(x_test)[:][:,1]
df_actual_predicted_KNN_1 = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_KNN_1, columns=['y_pred_proba_KNN_1'])], axis=1)
df_actual_predicted_KNN_1.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_KNN_1['label_1'], df_actual_predicted_KNN_1['y_pred_proba_KNN_1'])
auc_KNN_1 = roc_auc_score(df_actual_predicted_KNN_1['label_1'], df_actual_predicted_KNN_1['y_pred_proba_KNN_1'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for KNN 1', size = 10)
# plt.legend()


# # *KNN 3*
result_3_knn

y_pred_proba_KNN_3 = KNN_3.predict_proba(x_test)[:][:,1]
df_actual_predicted_KNN_3 = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_KNN_3, columns=['y_pred_proba_KNN_3'])], axis=1)
df_actual_predicted_KNN_3.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_KNN_3['label_1'], df_actual_predicted_KNN_3['y_pred_proba_KNN_3'])
auc_KNN_3 = roc_auc_score(df_actual_predicted_KNN_3['label_1'], df_actual_predicted_KNN_3['y_pred_proba_KNN_3'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for KNN 3', size = 10)
# plt.legend()


# # *KNN 5*

y_pred_proba_KNN_5 = KNN_5.predict_proba(x_test)[:][:,1]
df_actual_predicted_KNN_5 = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_KNN_5, columns=['y_pred_proba_KNN_5'])], axis=1)
df_actual_predicted_KNN_5.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_KNN_5['label_1'], df_actual_predicted_KNN_5['y_pred_proba_KNN_5'])
auc_KNN_5 = roc_auc_score(df_actual_predicted_KNN_5['label_1'], df_actual_predicted_KNN_5['y_pred_proba_KNN_5'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for KNN 5', size = 10)
# plt.legend()


# # *GBM 50*

y_pred_proba_GBM_50 = GBM_50.predict_proba(x_test)[:][:,1]
df_actual_predicted_GBM_50 = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_GBM_50, columns=['y_pred_proba_GBM_50'])], axis=1)
df_actual_predicted_GBM_50.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_GBM_50['label_1'], df_actual_predicted_GBM_50['y_pred_proba_GBM_50'])
auc_GBM_50 = roc_auc_score(df_actual_predicted_GBM_50['label_1'], df_actual_predicted_GBM_50['y_pred_proba_GBM_50'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for GBM_50', size = 10)
# plt.legend()


# # *GBM 100*
y_pred_proba_GBM_100 = GBM_100.predict_proba(x_test)[:][:,1]
df_actual_predicted_GBM_100 = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_GBM_100, columns=['y_pred_proba_GBM_100'])], axis=1)
df_actual_predicted_GBM_100.index = y_test.index
fpr, tpr, tr = metrics.roc_curve(df_actual_predicted_GBM_100['label_1'], df_actual_predicted_GBM_100['y_pred_proba_GBM_100'])
auc_GBM_100 = metrics.roc_auc_score(df_actual_predicted_GBM_100['label_1'], df_actual_predicted_GBM_100['y_pred_proba_GBM_100'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for GBM 100', size = 10)
# plt.legend()


# # *GBM 200*
y_pred_proba_GBM_200 = GBM_200.predict_proba(x_test)[:][:,1]
df_actual_predicted_GBM_200 = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_GBM_200, columns=['y_pred_proba_GBM_200'])], axis=1)
df_actual_predicted_GBM_200.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_GBM_200['label_1'], df_actual_predicted_GBM_200['y_pred_proba_GBM_200'])
auc_GBM_200 = roc_auc_score(df_actual_predicted_GBM_200['label_1'], df_actual_predicted_GBM_200['y_pred_proba_GBM_200'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for GBM 200', size = 10)
# plt.legend()


# # *Support Vector Classifier (kernel = RBF)*
y_pred_proba_RBF = RBF.predict_proba(x_test)[:][:,1]
df_actual_predicted_RBF = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_RBF, columns=['y_pred_proba_RBF'])], axis=1)
df_actual_predicted_RBF.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_RBF['label_1'], df_actual_predicted_RBF['y_pred_proba_RBF'])
auc_RBF = roc_auc_score(df_actual_predicted_RBF['label_1'], df_actual_predicted_RBF['y_pred_proba_RBF'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for RBF', size = 10)
# plt.legend()


# # *Support Vector Classifier (kernel = Linear)*

y_pred_proba_LINEAR = LINEAR.predict_proba(x_test)[:][:,1]
df_actual_predicted_LINEAR = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_LINEAR, columns=['y_pred_proba_LINEAR'])], axis=1)
df_actual_predicted_LINEAR.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_LINEAR['label_1'], df_actual_predicted_LINEAR['y_pred_proba_LINEAR'])
auc_LINEAR = roc_auc_score(df_actual_predicted_LINEAR['label_1'], df_actual_predicted_LINEAR['y_pred_proba_LINEAR'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Support Vector Classifier (Linear)', size = 10)
# plt.legend()


# # *XGBoost*
y_pred_proba_XGB = xgb_model.predict_proba(x_test)[:][:,1]
df_actual_predicted_XGB = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_XGB, columns=['y_pred_proba_XGB'])], axis=1)
df_actual_predicted_XGB.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_XGB['label_1'], df_actual_predicted_XGB['y_pred_proba_XGB'])
auc_XGB = roc_auc_score(df_actual_predicted_XGB['label_1'], df_actual_predicted_XGB['y_pred_proba_XGB'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for XGBoost', size = 10)
# plt.legend()


# # *Random Forest*
y_pred_proba_RNF = RNF.predict_proba(x_test)[:][:,1]
df_actual_predicted_RNF = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_RNF, columns=['y_pred_proba_RNF'])], axis=1)
df_actual_predicted_RNF.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_RNF['label_1'], df_actual_predicted_RNF['y_pred_proba_RNF'])
auc_RNF = roc_auc_score(df_actual_predicted_RNF['label_1'], df_actual_predicted_RNF['y_pred_proba_RNF'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for Random Forest', size = 10)
# plt.legend()


# # *Logestic Regression*
y_pred_proba_LR = LR.predict_proba(x_test)[:][:,1]
df_actual_predicted_LR = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_LR, columns=['y_pred_proba_LR'])], axis=1)
df_actual_predicted_LR.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_LR['label_1'], df_actual_predicted_LR['y_pred_proba_LR'])
auc_LR = roc_auc_score(df_actual_predicted_LR['label_1'], df_actual_predicted_LR['y_pred_proba_LR'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for Logistic Regression', size = 10)
# plt.legend()


# # *CatBoostClassifier*
catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, Y_train)
y_pred_catboost = catboost_model.predict(x_test)

print(classification_report(y_test, y_pred_test_CatBoost))

y_pred_proba_catboost = CatBoost.predict_proba(x_test)[:][:,1]
df_actual_predicted_catboost = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_catboost, columns=['y_pred_proba_catboost'])], axis=1)
df_actual_predicted_catboost.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_catboost['label_1'], df_actual_predicted_catboost['y_pred_proba_catboost'])
auc_catboost = roc_auc_score(df_actual_predicted_catboost['label_1'], df_actual_predicted_catboost['y_pred_proba_catboost'])
# plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
# plt.plot(fpr, fpr, linestyle = '--', color='k')
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curve for catboost', size = 10)
# plt.legend()


# # *LSTM*
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, SimpleRNN, Dropout

X_train = np.array(X_train)
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))


# Build the LSTM model
model = Sequential()
model.add(LSTM(units=256, input_shape=(1, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=32))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=50, batch_size=32)

score, accuracy = model.evaluate(x_test, y_test)

y_pred_proba_LSTM=model.predict(x_test)
# set threshold for converting probabilities to binary classes
threshold = 0.5

# # convert probabilities to binary classes using threshold
binary_preds_LSTM = np.where(y_pred_proba_LSTM >= threshold, 1, 0)

predict_classes=np.argmax(y_pred_proba_LSTM,axis=1)
df_actual_predicted_LSTM = pd.concat([pd.DataFrame(np.array(y_test), columns=['label_1']), pd.DataFrame(y_pred_proba_LSTM, columns=['y_pred_proba_LSTM'])], axis=1)
df_actual_predicted_LSTM.index = y_test.index
fpr, tpr, tr = roc_curve(df_actual_predicted_LSTM['label_1'], df_actual_predicted_LSTM['y_pred_proba_LSTM'])
auc = roc_auc_score(df_actual_predicted_LSTM['label_1'], df_actual_predicted_LSTM['y_pred_proba_LSTM'])

######################################################################################
#fit logistic regression model and plot ROC curve

fpr_LR, tpr_LR, _ = metrics.roc_curve(df_actual_predicted_LR['label_1'], df_actual_predicted_LR['y_pred_proba_LR'])
# auc_LR = round(auc_LR)
auc_LR = round(metrics.roc_auc_score(df_actual_predicted_LR['label_1'], df_actual_predicted_LR['y_pred_proba_LR']), 4)
plt.plot(fpr_LR,tpr_LR,label="Logistic Regression, AUC="+str(auc_LR), linewidth=3)

######################################################################################
#fit gradient boosted 100 model and plot ROC curve

fpr_GBM_50, tpr_GBM_50, _ = metrics.roc_curve(df_actual_predicted_GBM_50['label_1'], df_actual_predicted_GBM_50['y_pred_proba_GBM_50'])
# auc_GBM_50 = round(auc_GBM_50)
auc_GBM_50 = round(metrics.roc_auc_score(df_actual_predicted_GBM_50['label_1'], df_actual_predicted_GBM_50['y_pred_proba_GBM_50']), 4)
plt.plot(fpr_GBM_50,tpr_GBM_50,label="Gradient Boosting 50, AUC="+str(auc_GBM_50), linewidth=3)

######################################################################################
#fit gradient boosted 100 model and plot ROC curve

fpr_GBM_200, tpr_GBM_200, _ = metrics.roc_curve(df_actual_predicted_GBM_200['label_1'], df_actual_predicted_GBM_200['y_pred_proba_GBM_200'])
# auc_GBM_200 = round(auc_GBM_200)
auc_GBM_200 = round(metrics.roc_auc_score(df_actual_predicted_GBM_200['label_1'], df_actual_predicted_GBM_200['y_pred_proba_GBM_200']), 4)
plt.plot(fpr_GBM_200,tpr_GBM_200,label="Gradient Boosting 200, AUC="+str(auc_GBM_200), linewidth=3)


######################################################################################
#fit gradient boosted 50 model and plot ROC curve

fpr_GBM_100, tpr_GBM_100, _ = metrics.roc_curve(df_actual_predicted_GBM_100['label_1'], df_actual_predicted_GBM_100['y_pred_proba_GBM_100'])
auc_GBM_100 = round(auc_GBM_100)
# auc = round(metrics.roc_auc_score(df_actual_predicted_GBM_100['label_1'], df_actual_predicted_GBM_100['y_pred_proba_GBM_100']), 4)
plt.plot(fpr_GBM_100,tpr_GBM_100, label="Gradient Boosting 100, AUC="+str(auc_GBM_100), linewidth=3)


######################################################################################
#fit Knn 1 model and plot ROC curve

fpr_KNN_1, tpr_KNN_1, _ = metrics.roc_curve(df_actual_predicted_KNN_1['label_1'], df_actual_predicted_KNN_1['y_pred_proba_KNN_1'])
# auc_KNN_1 = round(auc_KNN_1)
auc_KNN_1 = round(metrics.roc_auc_score(df_actual_predicted_KNN_1['label_1'], df_actual_predicted_KNN_1['y_pred_proba_KNN_1']), 4)
plt.plot(fpr_KNN_1,tpr_KNN_1,label="Knn 1 , AUC="+str(auc_KNN_1), linewidth=3)


######################################################################################
#fit Knn 3 model and plot ROC curve

fpr_KNN_3, tpr_KNN_3, _ = metrics.roc_curve(df_actual_predicted_KNN_3['label_1'], df_actual_predicted_KNN_3['y_pred_proba_KNN_3'])
# auc_KNN_3 = round(auc_KNN_3)
auc_KNN_3 = round(metrics.roc_auc_score(df_actual_predicted_KNN_3['label_1'], df_actual_predicted_KNN_3['y_pred_proba_KNN_3']), 4)
plt.plot(fpr_KNN_3,tpr_KNN_3,label="Knn 3, AUC="+str(auc_KNN_1), linewidth=3)

######################################################################################
#fit Knn 5 model and plot ROC curve

fpr_KNN_5, tpr_KNN_5, _ = metrics.roc_curve(df_actual_predicted_KNN_5['label_1'], df_actual_predicted_KNN_5['y_pred_proba_KNN_5'])
# auc_KNN_5 = round(auc_KNN_5)
auc_KNN_5 = round(metrics.roc_auc_score(df_actual_predicted_KNN_5['label_1'], df_actual_predicted_KNN_5['y_pred_proba_KNN_5']), 4)
plt.plot(fpr_KNN_5,tpr_KNN_5,label="Knn 5, AUC="+str(auc_KNN_5), linewidth=3)


######################################################################################
#fit XGboost model and plot ROC curve

fpr_XGB, tpr_XGB, _ = metrics.roc_curve(df_actual_predicted_XGB['label_1'], df_actual_predicted_XGB['y_pred_proba_XGB'])
# auc_XGB = round(auc_XGB)
auc_XGB = round(metrics.roc_auc_score(df_actual_predicted_XGB['label_1'], df_actual_predicted_XGB['y_pred_proba_XGB']), 4)
plt.plot(fpr_XGB,tpr_XGB,label="XGBoost, AUC="+str(auc_XGB), linewidth=3)


######################################################################################
#fit Random Forest model and plot ROC curve

fpr_RNF, tpr_RNF, _ = metrics.roc_curve(df_actual_predicted_RNF['label_1'], df_actual_predicted_RNF['y_pred_proba_RNF'])
# auc_RNF = round(auc_RNF)
auc_RNF = round(metrics.roc_auc_score(df_actual_predicted_RNF['label_1'], df_actual_predicted_RNF['y_pred_proba_RNF']), 4)
plt.plot(fpr_RNF,tpr_RNF,label="Random Forest, AUC="+str(auc_RNF), linewidth=3)


######################################################################################
#fit CART model and plot ROC curve

fpr_CART, tpr_CART, _ = metrics.roc_curve(df_actual_predicted_CART['label_1'], df_actual_predicted_CART['y_pred_proba_CART'])
# auc_CART = round(auc_CART)
auc_CART = round(metrics.roc_auc_score(df_actual_predicted_CART['label_1'], df_actual_predicted_CART['y_pred_proba_CART']), 4)
plt.plot(fpr_CART,tpr_CART,label="Decision Tree, AUC="+str(auc_CART), linewidth=3)

######################################################################################
#fit catboost model and plot ROC curve

fpr_catboost, tpr_catboost, _ = metrics.roc_curve(df_actual_predicted_catboost['label_1'], df_actual_predicted_catboost['y_pred_proba_catboost'])
# auc_catboost = round(auc_catboost)
auc_catboost = round(metrics.roc_auc_score(df_actual_predicted_catboost['label_1'], df_actual_predicted_catboost['y_pred_proba_catboost']), 4)
plt.plot(fpr_catboost,tpr_catboost,label="Catboost, AUC="+str(auc_catboost), linewidth=3)

######################################################################################
#fit SVC model and plot ROC curve ----> default kernel is Gaussian Radial Basis Function (RBF)

fpr_RBF, tpr_RBF, _ = metrics.roc_curve(df_actual_predicted_RBF['label_1'], df_actual_predicted_RBF['y_pred_proba_RBF'])
# auc_RBF = round(auc_RBF)
auc_RBF = round(metrics.roc_auc_score(df_actual_predicted_RBF['label_1'], df_actual_predicted_RBF['y_pred_proba_RBF']), 4)
plt.plot(fpr_RBF,tpr_RBF,label="SVC - RFB, AUC="+str(auc_RBF), linewidth=3)

######################################################################################
#fit SVC model and plot ROC curve ----> default kernel is Linear

fpr_LINEAR, tpr_LINEAR, _ = metrics.roc_curve(df_actual_predicted_LINEAR['label_1'], df_actual_predicted_LINEAR['y_pred_proba_LINEAR'])
# auc_LINEAR = round(auc_LINEAR)
auc_LINEAR = round(metrics.roc_auc_score(df_actual_predicted_LINEAR['label_1'], df_actual_predicted_LINEAR['y_pred_proba_LINEAR']), 4)
plt.plot(fpr_LINEAR,tpr_LINEAR,label="SVC - LINEAR, AUC="+str(auc_LINEAR), linewidth=3)

#######################################################################################
#fit LSTM model and plot ROC curve ---->

fpr, tpr, _ = metrics.roc_curve(df_actual_predicted_LSTM['label_1'], df_actual_predicted_LSTM['y_pred_proba_LSTM'])
auc = round(metrics.roc_auc_score(df_actual_predicted_LSTM['label_1'], df_actual_predicted_LSTM['y_pred_proba_LSTM']), 4)
plt.plot(fpr,tpr,label="LSTM, AUC="+str(auc), linewidth=3)

def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(25, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()


plot_feature_importance(CatBoost.get_feature_importance(), X_DataFrame.columns, 'CATBOOST')

f1_score_CART = metrics.f1_score(y_test,y_pred_test_CART)
Accuracy_CART = metrics.accuracy_score(y_test,y_pred_test_CART)
Recall_CART = metrics.recall_score(y_test,y_pred_test_CART)
##########
error_rate_CART = 1 - Accuracy_CART
correctly_percentage_CART = Accuracy_CART * 100
incorrectly_percentage_CART = error_rate_CART * 100
##########
precision_CART = metrics.precision_score(y_test, y_pred_test_CART, average=None)
precision_CART_weighted = metrics.precision_score(y_test, y_pred_test_CART, average='weighted')
precision_CART_micro = metrics.precision_score(y_test, y_pred_test_CART, average='micro')
balanced_accuracy_CART = metrics.balanced_accuracy_score(y_test, y_pred_test_CART)
auc_CART=metrics.roc_auc_score(y_test, y_pred_test_CART)
# # print('f1-score for CART: {}'.format(f1_score_CART))
CART_df = [['CART', auc_CART, f1_score_CART, Accuracy_CART, precision_CART_weighted,
            precision_CART_micro, balanced_accuracy_CART, Recall_CART,
            correctly_percentage_CART, incorrectly_percentage_CART]]
CART_reults = pd.DataFrame(CART_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_KNN_1 = metrics.f1_score(y_test,y_pred_test_1_KNN)
Accuracy_KNN_1 = metrics.accuracy_score(y_test,y_pred_test_1_KNN)
Recall_KNN_1 = metrics.recall_score(y_test,y_pred_test_1_KNN)
##########
error_rate_KNN_1 = 1 - Accuracy_KNN_1
correctly_percentage_KNN_1 = Accuracy_KNN_1 * 100
incorrectly_percentage_KNN_1 = error_rate_KNN_1 * 100
##########
precision_KNN_1 = metrics.precision_score(y_test, y_pred_test_1_KNN, average=None)
precision_KNN_1_weighted = metrics.precision_score(y_test, y_pred_test_1_KNN, average='weighted')
precision_KNN_1_micro = metrics.precision_score(y_test, y_pred_test_1_KNN, average='micro')
balanced_accuracy_KNN_1 = metrics.balanced_accuracy_score(y_test, y_pred_test_1_KNN)
auc_KNN_1 = metrics.roc_auc_score(y_test, y_pred_test_1_KNN)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
KNN_1_df = [['KNN-1', auc_KNN_1, f1_score_KNN_1, Accuracy_KNN_1, precision_KNN_1_weighted,
            precision_KNN_1_micro, balanced_accuracy_KNN_1, Recall_KNN_1,
             correctly_percentage_KNN_1, incorrectly_percentage_KNN_1]]
KNN_1_reults = pd.DataFrame(KNN_1_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])

f1_score_KNN_3 = metrics.f1_score(y_test,y_pred_test_3_KNN)
Accuracy_KNN_3 = metrics.accuracy_score(y_test,y_pred_test_3_KNN)
Recall_KNN_3 = metrics.recall_score(y_test,y_pred_test_3_KNN)
##########
error_rate_KNN_3 = 1 - Accuracy_KNN_3
correctly_percentage_KNN_3 = Accuracy_KNN_3 * 100
incorrectly_percentage_KNN_3 = error_rate_KNN_3 * 100
##########
precision_KNN_3 = metrics.precision_score(y_test, y_pred_test_3_KNN, average=None)
precision_KNN_3_weighted = metrics.precision_score(y_test, y_pred_test_3_KNN, average='weighted')
precision_KNN_3_micro = metrics.precision_score(y_test, y_pred_test_3_KNN, average='micro')
balanced_accuracy_KNN_3= metrics.balanced_accuracy_score(y_test, y_pred_test_3_KNN)
auc_KNN_3 = metrics.roc_auc_score(y_test, y_pred_test_3_KNN)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
KNN_3_df = [['KNN-3', auc_KNN_3, f1_score_KNN_3, Accuracy_KNN_3, precision_KNN_3_weighted,
            precision_KNN_3_micro, balanced_accuracy_KNN_3, Recall_KNN_3,
             correctly_percentage_KNN_3, incorrectly_percentage_KNN_3]]
KNN_3_reults = pd.DataFrame(KNN_3_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_KNN_5 = metrics.f1_score(y_test,y_pred_test_5_KNN)
Accuracy_KNN_5 = metrics.accuracy_score(y_test,y_pred_test_5_KNN)
Recall_KNN_5 = metrics.recall_score(y_test,y_pred_test_5_KNN)
##########
error_rate_KNN_5 = 1 - Accuracy_KNN_5
correctly_percentage_KNN_5 = Accuracy_KNN_5 * 100
incorrectly_percentage_KNN_5 = error_rate_KNN_5 * 100
##########
precision_KNN_5 = metrics.precision_score(y_test, y_pred_test_5_KNN, average=None)
precision_KNN_5_weighted = metrics.precision_score(y_test, y_pred_test_5_KNN, average='weighted')
precision_KNN_5_micro = metrics.precision_score(y_test, y_pred_test_5_KNN, average='micro')
balanced_accuracy_KNN_5 = metrics.balanced_accuracy_score(y_test, y_pred_test_5_KNN)
auc_KNN_5 = metrics.roc_auc_score(y_test, y_pred_test_5_KNN)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
KNN_5_df = [['KNN-5', auc_KNN_5, f1_score_KNN_5, Accuracy_KNN_5, precision_KNN_5_weighted,
            precision_KNN_5_micro, balanced_accuracy_KNN_5, Recall_KNN_5,
             correctly_percentage_KNN_5, incorrectly_percentage_KNN_5]]
KNN_5_reults = pd.DataFrame(KNN_5_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])




f1_score_GBM_50 = metrics.f1_score(y_test,y_pred_test_GBM_50)
Accuracy_GBM_50 = metrics.accuracy_score(y_test,y_pred_test_GBM_50)
Recall_GBM_50 = metrics.recall_score(y_test,y_pred_test_GBM_50)
##########
error_rate_GBM_50 = 1 - Accuracy_GBM_50
correctly_percentage_GBM_50 = Accuracy_GBM_50 * 100
incorrectly_percentage_GBM_50 = error_rate_GBM_50 * 100
##########
precision_GBM_50 = metrics.precision_score(y_test, y_pred_test_GBM_50, average=None)
precision_GBM_50_weighted = metrics.precision_score(y_test, y_pred_test_GBM_50, average='weighted')
precision_GBM_50_micro = metrics.precision_score(y_test, y_pred_test_GBM_50, average='micro')
balanced_accuracy_GBM_50 = metrics.balanced_accuracy_score(y_test, y_pred_test_GBM_50)
auc_GBM_50 = metrics.roc_auc_score(y_test, y_pred_test_GBM_50)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
GBM_50_df = [['GBM_50', auc_GBM_50, f1_score_GBM_50, Accuracy_GBM_50, precision_GBM_50_weighted,
            precision_GBM_50_micro, balanced_accuracy_GBM_50, Recall_GBM_50,
              correctly_percentage_GBM_50, incorrectly_percentage_GBM_50]]
GBM_50_reults = pd.DataFrame(GBM_50_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_GBM_100 = metrics.f1_score(y_test,y_pred_test_GBM_100)
Accuracy_GBM_100 = metrics.accuracy_score(y_test,y_pred_test_GBM_100)
Recall_GBM_100 = metrics.recall_score(y_test,y_pred_test_GBM_100)
##########
error_rate_GBM_100 = 1 - Accuracy_GBM_100
correctly_percentage_GBM_100 = Accuracy_GBM_100 * 100
incorrectly_percentage_GBM_100 = error_rate_GBM_100 * 100
##########
precision_GBM_100 = metrics.precision_score(y_test, y_pred_test_GBM_100, average=None)
precision_GBM_100_weighted = metrics.precision_score(y_test, y_pred_test_GBM_100, average='weighted')
precision_GBM_100_micro = metrics.precision_score(y_test, y_pred_test_GBM_100, average='micro')
balanced_accuracy_GBM_100 = metrics.balanced_accuracy_score(y_test, y_pred_test_GBM_100)
auc_GBM_100 = metrics.roc_auc_score(y_test, y_pred_test_GBM_100)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
GBM_100_df = [['GBM_100', auc_GBM_100, f1_score_GBM_100, Accuracy_GBM_100, precision_GBM_100_weighted,
            precision_GBM_100_micro, balanced_accuracy_GBM_100, Recall_GBM_100,
              correctly_percentage_GBM_100, incorrectly_percentage_GBM_100]]
GBM_100_reults = pd.DataFrame(GBM_100_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])

f1_score_GBM_200 = metrics.f1_score(y_test,y_pred_test_GBM_200)
Accuracy_GBM_200 = metrics.accuracy_score(y_test,y_pred_test_GBM_200)
Recall_GBM_200 = metrics.recall_score(y_test,y_pred_test_GBM_200)
##########
error_rate_GBM_200 = 1 - Accuracy_GBM_200
correctly_percentage_GBM_200 = Accuracy_GBM_200 * 100
incorrectly_percentage_GBM_200 = error_rate_GBM_200 * 100
##########
precision_GBM_200 = metrics.precision_score(y_test, y_pred_test_GBM_200, average=None)
precision_GBM_200_weighted = metrics.precision_score(y_test, y_pred_test_GBM_200, average='weighted')
precision_GBM_200_micro = metrics.precision_score(y_test, y_pred_test_GBM_200, average='micro')
balanced_accuracy_GBM_200 = metrics.balanced_accuracy_score(y_test, y_pred_test_GBM_200)
auc_GBM_200 = metrics.roc_auc_score(y_test, y_pred_test_GBM_200)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
GBM_200_df = [['GBM_200', auc_GBM_200, f1_score_GBM_200, Accuracy_GBM_200, precision_GBM_200_weighted,
            precision_GBM_200_micro, balanced_accuracy_GBM_200, Recall_GBM_200,
              correctly_percentage_GBM_200, incorrectly_percentage_GBM_200]]
GBM_200_reults = pd.DataFrame(GBM_200_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])

f1_score_SVC_RBF = metrics.f1_score(y_test,y_pred_test_RBF)
Accuracy_SVC_RBF = metrics.accuracy_score(y_test,y_pred_test_RBF)
Recall_SVC_RBF = metrics.recall_score(y_test,y_pred_test_RBF)
##########
error_rate_SVC_RBF = 1 - Accuracy_SVC_RBF
correctly_percentage_SVC_RBF = Accuracy_SVC_RBF * 100
incorrectly_percentage_SVC_RBF = error_rate_SVC_RBF * 100
##########
precision_SVC_RBF = metrics.precision_score(y_test, y_pred_test_RBF, average=None)
precision_SVC_RBF_weighted = metrics.precision_score(y_test, y_pred_test_RBF, average='weighted')
precision_SVC_RBF_micro = metrics.precision_score(y_test, y_pred_test_RBF, average='micro')
balanced_accuracy_SVC_RBF = metrics.balanced_accuracy_score(y_test, y_pred_test_RBF)
auc_SVC_RBF = metrics.roc_auc_score(y_test, y_pred_test_RBF)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
SVC_RBF_df = [['SVC_RBF', auc_SVC_RBF, f1_score_SVC_RBF, Accuracy_SVC_RBF, precision_SVC_RBF_weighted,
            precision_SVC_RBF_micro, balanced_accuracy_SVC_RBF, Recall_SVC_RBF,
              correctly_percentage_SVC_RBF, incorrectly_percentage_SVC_RBF]]
SVC_RBF_reults = pd.DataFrame(SVC_RBF_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_SVC_Linear = metrics.f1_score(y_test,y_pred_test_LINEAR)
Accuracy_SVC_Linear = metrics.accuracy_score(y_test,y_pred_test_LINEAR)
Recall_SVC_Linear = metrics.recall_score(y_test,y_pred_test_LINEAR)
##########
error_rate_SVC_Linear = 1 - Accuracy_SVC_Linear
correctly_percentage_SVC_Linear = Accuracy_SVC_Linear * 100
incorrectly_percentage_SVC_Linear = error_rate_SVC_Linear * 100
##########
precision_SVC_Linear = metrics.precision_score(y_test, y_pred_test_LINEAR, average=None)
precision_SVC_Linear_weighted = metrics.precision_score(y_test, y_pred_test_LINEAR, average='weighted')
precision_SVC_Linear_micro = metrics.precision_score(y_test, y_pred_test_LINEAR, average='micro')
balanced_accuracy_SVC_Linear= metrics.balanced_accuracy_score(y_test, y_pred_test_LINEAR)
auc_SVC_Linear = metrics.roc_auc_score(y_test, y_pred_test_LINEAR)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
SVC_Linear_df = [['SVC_Linear', auc_SVC_Linear, f1_score_SVC_Linear, Accuracy_SVC_Linear, precision_SVC_Linear_weighted,
            precision_SVC_Linear_micro, balanced_accuracy_SVC_Linear, Recall_SVC_Linear,
                 correctly_percentage_SVC_Linear, incorrectly_percentage_SVC_Linear]]
SVC_Linear_results = pd.DataFrame(SVC_Linear_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_XGBoost = metrics.f1_score(y_test,y_pred_test_XGB)
Accuracy_XGBoost = metrics.accuracy_score(y_test,y_pred_test_XGB)
Recall_SVC_XGBoost = metrics.recall_score(y_test,y_pred_test_XGB)
##########
error_rate_XGBoost = 1 - Accuracy_XGBoost
correctly_percentage_XGBoost = Accuracy_XGBoost * 100
incorrectly_percentage_XGBoost = error_rate_XGBoost * 100
##########
precision_XGBoost = metrics.precision_score(y_test, y_pred_test_XGB, average=None)
precision_XGBoost_weighted = metrics.precision_score(y_test, y_pred_test_XGB, average='weighted')
precision_XGBoost_micro = metrics.precision_score(y_test, y_pred_test_XGB, average='micro')
balanced_accuracy_XGBoost = metrics.balanced_accuracy_score(y_test, y_pred_test_XGB)
auc_XGBoost = metrics.roc_auc_score(y_test, y_pred_test_XGB)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
XGBoost_df = [['XGBoost', auc_XGBoost, f1_score_XGBoost, Accuracy_XGBoost, precision_XGBoost_weighted,
            precision_XGBoost_micro, balanced_accuracy_XGBoost, Recall_SVC_XGBoost,
              correctly_percentage_XGBoost, incorrectly_percentage_XGBoost]]
XGBoost_reults = pd.DataFrame(XGBoost_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_RNF = metrics.f1_score(y_test,y_pred_test_RNF)
Accuracy_RNF = metrics.accuracy_score(y_test,y_pred_test_RNF)
Recall_SVC_RNF = metrics.recall_score(y_test,y_pred_test_RNF)
##########
error_rate_RNF = 1 - Accuracy_RNF
correctly_percentage_RNF = Accuracy_RNF * 100
incorrectly_percentage_RNF = error_rate_RNF * 100
##########
precision_RNF = metrics.precision_score(y_test, y_pred_test_RNF, average=None)
precision_RNF_weighted = metrics.precision_score(y_test, y_pred_test_RNF, average='weighted')
precision_RNF_micro = metrics.precision_score(y_test, y_pred_test_RNF, average='micro')
balanced_accuracy_RNF = metrics.balanced_accuracy_score(y_test, y_pred_test_RNF)
auc_RNF = metrics.roc_auc_score(y_test, y_pred_test_RNF)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
RNF_df = [['Random Forest', auc_RNF, f1_score_RNF, Accuracy_RNF, precision_RNF_weighted,
            precision_RNF_micro, balanced_accuracy_RNF, Recall_SVC_RNF,
          correctly_percentage_RNF, incorrectly_percentage_RNF]]
RNF_reults = pd.DataFrame(RNF_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_LR = metrics.f1_score(y_test,y_pred_test_LR)
Accuracy_LR = metrics.accuracy_score(y_test,y_pred_test_LR)
Recall_LR = metrics.recall_score(y_test,y_pred_test_LR)
##########
error_rate_LR = 1 - Accuracy_LR
correctly_percentage_LR = Accuracy_LR * 100
incorrectly_percentage_LR = error_rate_LR * 100
##########
precision_LR = metrics.precision_score(y_test, y_pred_test_LR, average=None)
precision_LR_weighted = metrics.precision_score(y_test, y_pred_test_LR, average='weighted')
precision_LR_micro = metrics.precision_score(y_test, y_pred_test_LR, average='micro')
balanced_accuracy_LR = metrics.balanced_accuracy_score(y_test, y_pred_test_LR)
auc_LR = metrics.roc_auc_score(y_test, y_pred_test_LR)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
LR_df = [['Logestic Regression', auc_LR, f1_score_LR, Accuracy_LR, precision_LR_weighted,
            precision_LR_micro, balanced_accuracy_LR, Recall_LR,
         correctly_percentage_LR, incorrectly_percentage_LR]]
LR_reults = pd.DataFrame(LR_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_CatBoost = metrics.f1_score(y_test,y_pred_test_CatBoost)
Accuracy_CatBoost = metrics.accuracy_score(y_test,y_pred_test_CatBoost)
Recall_CatBoost = metrics.recall_score(y_test,y_pred_test_CatBoost)
##########
error_rate_CatBoost = 1 - Accuracy_CatBoost
correctly_percentage_CatBoost = Accuracy_CatBoost * 100
incorrectly_percentage_CatBoost = error_rate_CatBoost * 100
##########
precision_CatBoost = metrics.precision_score(y_test, y_pred_test_CatBoost, average=None)
precision_CatBoost_weighted = metrics.precision_score(y_test, y_pred_test_CatBoost, average='weighted')
precision_CatBoost_micro = metrics.precision_score(y_test, y_pred_test_CatBoost, average='micro')
balanced_accuracy_CatBoost = metrics.balanced_accuracy_score(y_test, y_pred_test_CatBoost)
auc_CatBoost = metrics.roc_auc_score(y_test, y_pred_test_CatBoost)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
CatBoost_df = [['CatBoost', auc_CatBoost, f1_score_CatBoost, Accuracy_CatBoost, precision_CatBoost_weighted,
            precision_CatBoost_micro, balanced_accuracy_CatBoost, Recall_CatBoost,
               correctly_percentage_CatBoost, incorrectly_percentage_CatBoost]]
CatBoost_reults = pd.DataFrame(CatBoost_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])


f1_score_LSTM = f1_score(y_test,binary_preds_LSTM)
Accuracy_LSTM = accuracy_score(y_test,binary_preds_LSTM)
Recall_LSTM = metrics.recall_score(y_test,binary_preds_LSTM)
##########
error_rate_LSTM = 1 - Accuracy_LSTM
correctly_percentage_LSTM = Accuracy_LSTM * 100
incorrectly_percentage_LSTM = error_rate_LSTM * 100
##########
precision_LSTM = precision_score(y_test, binary_preds_LSTM, average=None)
precision_LSTM_weighted = precision_score(y_test, binary_preds_LSTM, average='weighted')
precision_LSTM_micro = precision_score(y_test, binary_preds_LSTM, average='micro')
balanced_accuracy_LSTM = balanced_accuracy_score(y_test, binary_preds_LSTM)
auc_LSTM = metrics.roc_auc_score(y_test, binary_preds_LSTM)
# print('f1-score for KNN-1: {}'.format(f1_score(y_test,y_pred_test_1_KNN)))
LSTM_df = [['LSTM', auc_LSTM, f1_score_LSTM, Accuracy_LSTM, precision_LSTM_weighted,
            precision_LSTM_micro, balanced_accuracy_LSTM, Recall_LSTM,
               correctly_percentage_LSTM, incorrectly_percentage_LSTM]]
LSTM_reults = pd.DataFrame(LSTM_df, columns=['Model name', 'AUC', 'f1_score', 'accuracy',
                                             'precision_weighted', 'precision_micro', 'balanced_accuracy', 'Recall',
                                            'correctly_percentage', 'incorrectly_percentage'])

result_Normal = pd.concat(frames)

result_Normal = result_Normal.round(2)
result_Normal = result_Normal.reset_index(drop=True)
print(result_Normal)

final_results_based_on_AUC_Unsup_Telco = result_Normal.sort_values(by=['AUC'], ascending=False)
# final_results_based_on_AUC_Unsup_Telco = final_results_based_on_AUC_Unsup_Telco.reset_index(drop=True)
final_results_based_on_AUC_Unsup_Telco = final_results_based_on_AUC_Unsup_Telco.round(2)
final_results_based_on_AUC_Unsup_Telco.to_csv('two_phase_resampling_telco.csv')
print(final_results_based_on_AUC_Unsup_Telco)






