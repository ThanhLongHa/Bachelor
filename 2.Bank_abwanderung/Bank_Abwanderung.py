#Quelle: https://www.kaggle.com/code/ahmtcnbs/customer-churn-analysis-xgb-86#Data-Preprocessing
import os
import time
start_time = time.time()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
pid = os.getpid()
print("ProcessID: ",pid)
#Import
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,precision_score,recall_score,f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#Preprocessing
df = pd.read_csv("D:/Uni/Bachelor/ProjectData/Churn/Churn_Modelling.csv")
df.drop(['CustomerId','Surname'],inplace=True,axis=1)
le = LabelEncoder()
df['Geography'] = le.fit_transform(df['Geography'])
df['Gender'] = le.fit_transform(df['Gender'])
# EstimatedSalary Klassifizieren
df.loc[df['EstimatedSalary'] <= 40007, 'EstimatedSalary'] = 0
df.loc[(df['EstimatedSalary'] > 40007) & (df['EstimatedSalary'] <= 80003), 'EstimatedSalary'] = 1
df.loc[(df['EstimatedSalary'] > 80003) & (df['EstimatedSalary'] <= 120000), 'EstimatedSalary'] = 2
df.loc[(df['EstimatedSalary'] > 120000) & (df['EstimatedSalary'] <= 159996), 'EstimatedSalary'] = 3
df.loc[df['EstimatedSalary'] > 159996, 'EstimatedSalary'] = 4
# Balance klassifizieren
df.loc[df['Balance'] <= 0, 'Balance'] = 0
df.loc[(df['Balance'] > 0) & (df['Balance'] <= 251), 'Balance'] = 1
df.loc[(df['Balance'] > 251) & (df['Balance'] <= 50179), 'Balance'] = 2
df.loc[(df['Balance'] > 50179) & (df['Balance'] <= 100359), 'Balance'] = 3
df.loc[(df['Balance'] > 100359) & (df['Balance'] <= 150538), 'Balance'] = 4
df.loc[(df['Balance'] > 150538) & (df['Balance'] <= 200718), 'Balance'] = 5
df.loc[(df['Balance'] > 200718) & (df['Balance'] <= 250000), 'Balance'] = 6
#Credit score klassifizieren
df.loc[df['CreditScore'] <= 450, 'CreditScore'] = 0
df.loc[(df['CreditScore'] > 450) & (df['CreditScore'] <= 550), 'CreditScore'] = 1
df.loc[(df['CreditScore'] > 550) & (df['CreditScore'] <= 650), 'CreditScore'] = 2
df.loc[(df['CreditScore'] > 650) & (df['CreditScore'] <= 750), 'CreditScore'] = 3
df.loc[(df['CreditScore'] > 750) & (df['CreditScore'] <= 850), 'CreditScore'] = 4
df.loc[(df['CreditScore'] > 850), 'CreditScore'] = 5

X = df.drop('Exited',axis=1)
y = df['Exited']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state=42)

#Model deployment
models = []
models.append(['Logistic Regression',LogisticRegression(random_state=0)])
models.append(['SVM',SVC(random_state=0)])
models.append(['KNeigbors',KNeighborsClassifier()])
models.append(['GaussianNB',GaussianNB()])
models.append(['DecisionTree',DecisionTreeClassifier(random_state=0)])
models.append(['RandomForest',RandomForestClassifier(random_state=0)])
lst_1 = []
for m in range(len(models)):
    lst_2 = []
    model = models[m][1]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test,y_pred)
    accuracies = cross_val_score(estimator= model, X = X_train,y = y_train, cv=10)

# k-fOLD Validation
    roc = roc_auc_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1 = f1_score(y_test,y_pred)
    print(models[m][0],':')
    print(cm)
    print('')
    print('Accuracy Score: ', accuracy_score(y_test, y_pred))
    print('')
    print('Standard Deviation: {:.2f} %'.format(accuracies.std() * 100))
    print('')
    print('ROC AUC Score: {:.2f} %'.format(roc))
    print('')
    print('Precision: {:.2f} %'.format(precision))
    print('')
    print('Recall: {:.2f} %'.format(recall))
    print('')
    print('F1 Score: {:.2f} %'.format(f1))
    print('-' * 40)
    print('')
    lst_2.append(models[m][0])
    lst_2.append(accuracy_score(y_test,y_pred)*100)
    lst_2.append(accuracies.mean()*100)
    lst_2.append(accuracies.std()*100)
    lst_2.append(roc)
    lst_2.append(precision)
    lst_2.append(recall)
    lst_2.append(f1)
    lst_1.append(lst_2)
end_time= time.time()
print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))
print("Dauer: ",(end_time-start_time))