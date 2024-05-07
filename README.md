# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Start the program

2.Import the python pandas library as pd

3.Read the contents of the Spam csv file

4.Display the first 5 rows of the dataset using head()

5.Assign x as v1 values and y as v2 values

6.From sklearn library select the feature extraction and import CountVectorizer

7.CountVectorizer will convert the Text to Numerical Data

8.From sklearn library import Support Vector Classifier (ie. SVC)

9.Predict the x_test using SVC

10.Print the accuracy of the SVM Model 11.Stop the program 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRAJAN P
RegisterNumber: 212223240121
*/
import chardet
file='/content/spam.csv'
with open(file, 'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("/content/spam.csv",encoding = 'Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values

y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
1.Result output:
![243171161-3eab037b-6809-422e-873d-f9ed78e8a1ad](https://github.com/PRAJAN-23013995/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150313345/d585280e-8636-48ed-98b1-3f23f31ea6ad)

2.data.head():
![243171274-bef21527-e9ef-4e71-bfa4-15658495faa7](https://github.com/PRAJAN-23013995/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150313345/761808aa-3fdb-4df6-b795-eea766254f82)

3.data.info():
![243171304-ea4dfc15-dd68-4050-b0d9-c601412d8074](https://github.com/PRAJAN-23013995/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150313345/ac610a27-060b-421f-9a33-686864362931)

data.isnull().sum():
![243171358-ccbf5240-2004-4419-a299-71feccb702bb](https://github.com/PRAJAN-23013995/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150313345/d4673c4a-3b8e-4365-9589-59e688d83fcb)

5.Y_prediction value:
![243171386-c4cb968d-d084-4389-9350-d6632f19b874](https://github.com/PRAJAN-23013995/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150313345/a4ac2a99-13f8-4982-bec3-607b3e2741c4)

6.Accuracy:
![243171390-e8577b82-305d-43be-ac7b-5e830b680157](https://github.com/PRAJAN-23013995/Implementation-of-SVM-For-Spam-Mail-Detection/assets/150313345/3e1e89bc-1425-4d12-82b5-c389226e0ef2)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
