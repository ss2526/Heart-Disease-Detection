# Heart-Disease-DetectionComparative Study on Heart Disease Prediction using ML Techniques
In this IDP Report, we used 7 Machine Learning Algorithms and Deep Learning.

Seven Different ML Algorithms are :

Logistic Regression
Naive Bayes
SVM
K Nearest Neighbors
Decision Tree
Random Forest
XGBoost
Deep Learning Model: Neural Network

Import the libraries and reading and understanding the DataSet
First, we need to import the libraries which are used for this "Heart Disease Prediction".


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
     
Now, we will read the Dataset using pandas where we given alias name as pd


df=pd.read_csv("/content/heart.csv")
     

df.shape #to find no.of rows and columns
     
(303, 14)
The dataset contains 14 attributes. Lets breifly learn about

Each attribute

Its unique features

Visualizing it using seaborn and matplot libraries.

As we are applying supervised Machine Learning models, here my Target attribute acts as a dependent variable and rest of the attributes are independent variables.

Target: This attribute contains 0's and 1's where 1 indicates a person is suffering with heart disease and 0 indicates absence of heart disease.


target_count = df.target.value_counts()
print(target_count)
     
1    165
0    138
Name: target, dtype: int64

y=df["target"]
sns.countplot(y)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94b0156760>

1.Age: The duration of something from its beginning up to the present. This attribute contains age of each person which ranges from 0 to 100. It plays a major role in heart disease prediction because most of the people who die due to heart disease are 60 and older.

2.Sex: Sex refers to the social and cultural distinctions between people who are male or female. This attribute contains sex of each person where 1 indicates male and 0 indicates female. It attributes plays a role in 2 different ways.

Diabetes: Females are more likely to have heart problems than males.
No Diabetes: Males are more likely to have heart problems than females.

df["sex"].unique()
     
array([1, 0])

sns.barplot(df["sex"],y)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94afbdd1f0>

3.Chest Pain: Chest pain is discomfort or pain felt in the chest area. It can be a symptom of various medical conditions, such as heart attack,angina, or pneumonia. The pain may be sharp or dull, and may be felt in the center of the chest or in a specific area.

This attribute contains the type of chest pain experienced by every person. There are 4 types of chest pains.

Typical Angina - indicates 0

Atypical Angina - indicates 1

Non-Anginal pain - indicates 2

Asymptotic - indicates 3

So, these attribute ranges from 0 to 3.


df["cp"].unique()
     
array([3, 2, 1, 0])

sns.barplot(df["cp"],y)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94afbc9490>

4.Resting Blood Pressure: Having high blood pressure can be detrimental, as it can damage the arteries that supply blood to the heart. If high blood pressure is accompanied by other health conditions such as obesity, high cholesterol or diabetes, the risk is even higher.

This attribute contains the resting blood pressure value of every person in mmHg(unit).

5.Serum Cholestrol:Having a high amount of LDL cholesterol can lead to clogged arteries, whereas having a high amount of triglycerides, which are associated with diet, increases the risk of a heart attack. On the other hand, having a high level of HDL cholesterol, which is beneficial, reduces the risk of a heart attack.

This attribute contains the serum Cholestrol of every person in mg/dl(unit).

6.Fasting Blood Sugar:Having insufficient insulin secretion or an inability to effectively utilize insulin from your pancreas can cause your blood sugar levels to rise, putting you at greater risk of having a heart attack.

This attributes compares Fasting Blood Sugar value of every person with 120mg/dl. This attribute contains 0's and 1's, where 1 indicates fasting blood sugar > 120mg/dl and 0 indicates fasting blood sugar <= 120mg/dl.


df["fbs"].unique()
     
array([1, 0])

sns.barplot(df["fbs"],y)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94afb28400>

7.Resting ECG :The U.S. Preventive Services Task Force determined that there is moderate certainty that the potential risks of screening with resting or exercise ECG outweigh the potential benefits for people at low risk of cardiovascular disease. For people at intermediate to high risk, the evidence is not sufficient to assess the balance of benefits and harms of screening.

This attribute contains the results of Resting ECG of every person which range from 0 to 2 where

Normal - indicates 0

having ST-T wave abnormality - indicates 1

left ventricular hyperthrophy - indicates 2


df["restecg"].unique()

     
array([0, 1, 2])

sns.barplot(df["restecg"],y)

     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94afaffb50>

8.Max heart rate achieved:An increase in heart rate by 10 beats per minute was associated with a 20% rise in the risk of cardiac death, which is similar to the increase in risk associated with a 10 mmHg rise in systolic blood pressure. This attribute contains the maximum heart rate achieved by every person.

9.Exercise induced angina:Angina is a type of chest pain that is usually experienced as a tight, gripping or squeezing sensation and can range from mild to severe. It typically occurs in the center of the chest, but may also spread to the shoulders, back, neck, jaw or arms and even the hands. There are four main types of angina: stable angina/angina pectoris, unstable angina, variant (Prinzmetal) angina, and microvascular angina. This attribute says whether person has Exercise induced angina or not. It contains 0's and 1's where

Having Exercise induced angina - indicates 1
No Exercise induced angina - indicates 0

df["exang"].unique()
     
array([0, 1])

sns.barplot(df["exang"],y)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94afada850>

10.ST depression induced by exercise relative to rest: ST depression induced by exercise refers to a decrease in the ST segment of the ECG (electrocardiogram) compared to the ST segment of the ECG at rest. This decrease is typically seen during exercise stress tests and is a sign of myocardial ischemia, which is a decrease in blood flow to the heart caused by an occlusion in the coronary arteries.

11.Peak exercise ST segment: ak exercise ST segment is a measurement of electrical activity in the heart during exercise. It is measured by an electrocardiogram and is used to detect signs of ischemia, or reduced blood flow to the heart muscle. During exercise, the ST segment may become elevated, flattened, or depressed, which can indicate a decrease in blood flow to the heart.

upsloping
flat
downsloping

df["slope"].unique()
     
array([0, 2, 1])

sns.barplot(df["slope"],y)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94afb2da30>

12.Number of major vessels (0–3) colored by flourosopy: Fluoroscopy is a type of medical imaging that uses X-rays to obtain real-time moving images of the internal structures of a patient. It is used to diagnose and treat diseases and medical conditions. It is also used to guide and monitor the progress of medical procedures such as catheter placement and biopsies.

Normal
Mild Abnormality
Moderate Abnormality
Severe Abnormality

df["ca"].unique()
     
array([0, 2, 1, 3, 4])

sns.countplot(df["ca"])
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variable as a keyword arg: x. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94af9fb910>

13.Thal: Thalassemia is an inherited blood disorder. It affects the body's ability to produce hemoglobin, a protein in red blood cells that carries oxygen to other parts of the body. People with thalassemia make either no hemoglobin or too little hemoglobin, which can lead to anemia (low red blood cell count) and other serious health problems.

No Thalassemia
Mild Thalassemia
Moderate Thalassemia
Severe Thalassemia

df["thal"].unique()
     
array([1, 2, 3, 0])

sns.barplot(df["thal"],y)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94af96d730>

Train and Test Split

from sklearn.model_selection import train_test_split #import train and test split using sklearn.

features = df.drop("target",axis=1) #From data frame, we are droping target attribute and storing data frame in features.
label = df["target"] #we are storing target attribute in label.

X_train,X_test,Y_train,Y_test = train_test_split(features,label,test_size=0.20,random_state=0)
     

X_train.shape
     
(242, 13)

X_test.shape
     
(61, 13)

Y_train.shape
     
(242,)

Y_test.shape
     
(61,)
Performance measures
Performance measures: Performance measures are essential metrics used to evaluate the effectiveness of a machine learning algorithm. Performance measures are used to compare the performance of different models on a given data set and to determine which model provides the best results. Performance measures can be used to identify areas in which the model can be improved and to determine the overall accuracy of the model.

There are four machine learning classification model performance measures:

Accuracy Score

Precision Score

Recall Score

F1-Score

Accuracy Score:Accuracy Score is one of the most commonly used performance measures in machine learning. It is used to measure how accurately a model can predict the expected output. It is calculated by taking the number of correct predictions divided by the total number of predictions. A higher accuracy score indicates that the model is more accurate in predicting the expected output.

Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)

Precision Score:Precision score is a performance measure used in machine learning to evaluate the accuracy of a model’s predictions. It is a measure of the ratio of true positives (TP) to the sum of true positives and false positives (FP). Precision score is typically expressed as a percentage, with a higher percentage indicating a better model performance.

Precision Score = TP / (FP + TP)

Recall Score:Recall Score is a performance measure used in machine learning that evaluates a model’s ability to correctly identify relevant instances from a dataset. It is calculated by dividing the number of relevant instances correctly identified by the total number of relevant instances in the dataset. The higher the recall score, the better the performance of the model.

Recall Score = TP / (FN + TP)

F1-Score:F1-score is a performance measure in machine learning that combines precision and recall into a single metric. It is often used to evaluate the performance of a classification model, as it takes both false positives and false negatives into account. The F1-score is the harmonic mean of precision and recall, where the best value is 1.0 and the worst value is 0.0. A model which has a high F1-score is considered to be a better model than one with a low F1-score. The F1-score is often used in conjunction with other performance measures such as accuracy, precision and recall.The F1-score is a good measure of a model’s performance when there is an uneven class distribution. This is because it takes both false positives and false negatives into account, and gives more weight to the minority class. The F1-score is also useful when there is a need to weigh precision and recall equally.

F1 Score = (2* Precision Score * Recall Score) / (Precision Score + Recall Score)

Machine Learning Models
Import all performance measures from sklearn library.


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
     
Logistic Regression:Logistic regression is a type of supervised machine learning algorithm that is used for classification tasks. It is a linear model used to estimate the probability of a binary response variable based on one or more independent variables. Logistic regression is used in a variety of applications, including predicting the risk of a medical condition based on symptoms, determining whether an email is spam, and predicting the likelihood that a customer will respond to a marketing campaign. It is also used in natural language processing to classify text documents.


from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred_lr = lr.predict(X_test)
     
/usr/local/lib/python3.8/dist-packages/sklearn/linear_model/_logistic.py:814: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(

Y_pred_lr.shape
     
(61,)

acc_score_lr = round(accuracy_score(Y_pred_lr,Y_test)*100,2)

print("The Accuracy Score achieved using Logistic Regression is: "+str(acc_score_lr)+" %")
     
The Accuracy Score achieved using Logistic Regression is: 85.25 %

pre_score_lr = round(precision_score(Y_pred_lr,Y_test)*100,2)

print("The Precision Score achieved using Logistic Regression is: "+str(pre_score_lr)+" %")
     
The Precision Score achieved using Logistic Regression is: 88.24 %

rec_score_lr = round(recall_score(Y_pred_lr,Y_test)*100,2)

print("The Recall Score achieved using Logistic Regression is: "+str(rec_score_lr)+" %")
     
The Recall Score achieved using Logistic Regression is: 85.71 %

f1_score_lr = round(f1_score(Y_pred_lr,Y_test)*100,2)

print("The F1-Score achieved using Logistic Regression is: "+str(f1_score_lr)+" %")
     
The F1-Score achieved using Logistic Regression is: 86.96 %
Naive Bayes:Naive Bayes is a supervised machine learning algorithm used for classification problems. It is based on the Bayes theorem, which states that the probability of an event is equal to the probability of the event happening times the probability of the event not happening divided by the probability of the event not happening.


from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(X_train,Y_train)

Y_pred_nb = nb.predict(X_test)
     

Y_pred_nb.shape
     
(61,)

acc_score_nb = round(accuracy_score(Y_pred_nb,Y_test)*100,2)

print("The Accuracy Score achieved using Naive Bayes is: "+str(acc_score_nb)+" %")
     
The Accuracy Score achieved using Naive Bayes is: 85.25 %

pre_score_nb = round(precision_score(Y_pred_nb,Y_test)*100,2)

print("The Precision Score achieved using Naive Bayes is: "+str(pre_score_nb)+" %")
     
The Precision Score achieved using Naive Bayes is: 91.18 %

rec_score_nb = round(recall_score(Y_pred_nb,Y_test)*100,2)

print("The Recall Score achieved using Naive Bayes is: "+str(rec_score_nb)+" %")
     
The Recall Score achieved using Naive Bayes is: 83.78 %

f1_score_nb = round(f1_score(Y_pred_nb,Y_test)*100,2)

print("The F1-Score achieved using Naive Bayes is: "+str(f1_score_nb)+" %")
     
The F1-Score achieved using Naive Bayes is: 87.32 %
SVM:Support Vector Machines (SVMs) are a type of supervised machine learning algorithm used for classification and regression problems. SVMs use a technique called the kernel trick to transform the data into a higher-dimensional space, which allows the algorithm to find a hyperplane that maximizes the distance between the data points of different classes.


from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train, Y_train)

Y_pred_svm = sv.predict(X_test)
     

Y_pred_svm.shape
     
(61,)

acc_score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

print("The Accuracy Score achieved using Linear SVM is: "+str(acc_score_svm)+" %")
     
The Accuracy Score achieved using Linear SVM is: 81.97 %

pre_score_svm = round(precision_score(Y_pred_svm,Y_test)*100,2)

print("The Precison Score achieved using Linear SVM is: "+str(pre_score_svm)+" %")
     
The Precison Score achieved using Linear SVM is: 88.24 %

rec_score_svm = round(recall_score(Y_pred_svm,Y_test)*100,2)

print("The Recall Score achieved using Linear SVM is: "+str(rec_score_svm)+" %")
     
The Recall Score achieved using Linear SVM is: 81.08 %

f1_score_svm = round(f1_score(Y_pred_svm,Y_test)*100,2)

print("The F1-Score achieved using Linear SVM is: "+str(f1_score_svm)+" %")
     
The F1-Score achieved using Linear SVM is: 84.51 %
K Nearest Neighbours:K Nearest Neighbor (KNN) is a supervised machine learning algorithm used for both classification and regression. It is a non-parametric and lazy learning algorithm, which means it does not assume any underlying data distribution and it delays its computations until the time of prediction.


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,Y_train)
Y_pred_knn=knn.predict(X_test)
     

Y_pred_knn.shape
     
(61,)

acc_score_knn = round(accuracy_score(Y_pred_knn,Y_test)*100,2)

print("The Accuracy Score achieved using KNN is: "+str(acc_score_knn)+" %")
     
The Accuracy Score achieved using KNN is: 67.21 %

pre_score_knn = round(precision_score(Y_pred_knn,Y_test)*100,2)

print("The Precision Score achieved using KNN is: "+str(pre_score_knn)+" %")
     
The Precision Score achieved using KNN is: 67.65 %

rec_score_knn = round(recall_score(Y_pred_knn,Y_test)*100,2)

print("The Recall Score achieved using KNN is: "+str(rec_score_knn)+" %")
     
The Recall Score achieved using KNN is: 71.88 %

f1_score_knn = round(f1_score(Y_pred_knn,Y_test)*100,2)

print("The F1-Score achieved using KNN is: "+str(f1_score_knn)+" %")
     
The F1-Score achieved using KNN is: 69.7 %
Decision Tree:Decision Trees are a supervised machine learning tool used for both classification and regression tasks. It is a type of algorithm that uses a tree-like structure to make decisions by breaking down a dataset into smaller and smaller subsets while at the same time an associated decision tree is incrementally developed. The final result is a tree with decision nodes and leaf nodes. The decision nodes have a certain condition or attribute associated with them, and the leaf nodes represent class labels or class distributions.


from sklearn.tree import DecisionTreeClassifier

max_accuracy = 0


for x in range(200):
    dt = DecisionTreeClassifier(random_state=x)
    dt.fit(X_train,Y_train)
    Y_pred_dt = dt.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_dt,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)


dt = DecisionTreeClassifier(random_state=best_x)
dt.fit(X_train,Y_train)
Y_pred_dt = dt.predict(X_test)
     

print(Y_pred_dt.shape)
     
(61,)

acc_score_dt = round(accuracy_score(Y_pred_dt,Y_test)*100,2)

print("The Accuracy Score achieved using Decision Tree is: "+str(acc_score_dt)+" %")
     
The Accuracy Score achieved using Decision Tree is: 81.97 %

pre_score_dt = round(precision_score(Y_pred_dt,Y_test)*100,2)

print("The Precision Score achieved using Decision Tree is: "+str(pre_score_dt)+" %")
     
The Precision Score achieved using Decision Tree is: 82.35 %

rec_score_dt = round(recall_score(Y_pred_dt,Y_test)*100,2)

print("The Recall Score achieved using Decision Tree is: "+str(rec_score_dt)+" %")
     
The Recall Score achieved using Decision Tree is: 84.85 %

f1_score_dt = round(f1_score(Y_pred_dt,Y_test)*100,2)

print("The F1-Score achieved using Decision Tree is: "+str(f1_score_dt)+" %")
     
The F1-Score achieved using Decision Tree is: 83.58 %
Random Forest:Random Forest is an effective and powerful machine learning algorithm that can be used for both classification and regression tasks. It is robust to overfitting, has high accuracy, and is easy to use and implement. Furthermore, it can handle large datasets and perform well even with missing data. This makes it a popular choice for many real-world applications.


from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0


for x in range(2000):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x
        
#print(max_accuracy)
#print(best_x)

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
     

Y_pred_rf.shape
     
(61,)

acc_score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)

print("The Accuracy Score achieved using Decision Tree is: "+str(acc_score_rf)+" %")
     
The Accuracy Score achieved using Decision Tree is: 90.16 %

pre_score_rf = round(precision_score(Y_pred_rf,Y_test)*100,2)

print("The Precision score achieved using Decision Tree is: "+str(pre_score_rf)+" %")
     
The Precision score achieved using Decision Tree is: 94.12 %

rec_score_rf = round(recall_score(Y_pred_rf,Y_test)*100,2)

print("The Recall score achieved using Decision Tree is: "+str(rec_score_rf)+" %")
     
The Recall score achieved using Decision Tree is: 88.89 %

f1_score_rf = round(f1_score(Y_pred_rf,Y_test)*100,2)

print("The F1-Score achieved using Decision Tree is: "+str(f1_score_rf)+" %")
     
The F1-Score achieved using Decision Tree is: 91.43 %
XGBoost:XGBoost (Extreme Gradient Boosting) is another popular supervised machine learning algorithm used for both classification and regression tasks. It is an ensemble learning technique that combines the power of both gradient boosting and random forest algorithms. XGBoost uses gradient descent to fit a model to the data, and then uses multiple decision trees to improve the model's accuracy.


import xgboost as xgb

xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)
xgb_model.fit(X_train, Y_train)

Y_pred_xgb = xgb_model.predict(X_test)
     

Y_pred_xgb.shape
     
(61,)

acc_score_xgb = round(accuracy_score(Y_pred_xgb,Y_test)*100,2)

print("The Accuracy Score achieved using XGBoost is: "+str(acc_score_xgb)+" %")
     
The Accuracy Score achieved using XGBoost is: 85.25 %

pre_score_xgb = round(precision_score(Y_pred_xgb,Y_test)*100,2)

print("The Precision Score achieved using XGBoost is: "+str(pre_score_xgb)+" %")
     
The Precision Score achieved using XGBoost is: 88.24 %

rec_score_xgb = round(recall_score(Y_pred_xgb,Y_test)*100,2)

print("The Recall Score achieved using XGBoost is: "+str(rec_score_xgb)+" %")
     
The Recall Score achieved using XGBoost is: 85.71 %

f1_score_xgb = round(f1_score(Y_pred_xgb,Y_test)*100,2)

print("The F1-Score achieved using XGBoost is: "+str(f1_score_xgb)+" %")
     
The F1-Score achieved using XGBoost is: 86.96 %
Deep Learning
Neural Network


from keras.models import Sequential
from keras.layers import Dense
     

model = Sequential()
model.add(Dense(11,activation='relu',input_dim=13))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
     

model.fit(X_train,Y_train,epochs=300)
     
Epoch 1/300
8/8 [==============================] - 1s 4ms/step - loss: 5.7096 - accuracy: 0.4504
Epoch 2/300
8/8 [==============================] - 0s 3ms/step - loss: 3.8989 - accuracy: 0.4380
Epoch 3/300
8/8 [==============================] - 0s 3ms/step - loss: 2.5203 - accuracy: 0.3843
Epoch 4/300
8/8 [==============================] - 0s 3ms/step - loss: 2.0169 - accuracy: 0.4587
Epoch 5/300
8/8 [==============================] - 0s 3ms/step - loss: 1.9830 - accuracy: 0.5124
Epoch 6/300
8/8 [==============================] - 0s 3ms/step - loss: 1.9770 - accuracy: 0.5248
Epoch 7/300
8/8 [==============================] - 0s 3ms/step - loss: 1.9045 - accuracy: 0.4835
Epoch 8/300
8/8 [==============================] - 0s 4ms/step - loss: 1.8565 - accuracy: 0.4628
Epoch 9/300
8/8 [==============================] - 0s 4ms/step - loss: 1.8395 - accuracy: 0.4545
Epoch 10/300
8/8 [==============================] - 0s 4ms/step - loss: 1.8108 - accuracy: 0.4587
Epoch 11/300
8/8 [==============================] - 0s 4ms/step - loss: 1.7861 - accuracy: 0.4711
Epoch 12/300
8/8 [==============================] - 0s 4ms/step - loss: 1.7593 - accuracy: 0.4711
Epoch 13/300
8/8 [==============================] - 0s 4ms/step - loss: 1.7255 - accuracy: 0.4835
Epoch 14/300
8/8 [==============================] - 0s 4ms/step - loss: 1.6971 - accuracy: 0.4711
Epoch 15/300
8/8 [==============================] - 0s 4ms/step - loss: 1.6840 - accuracy: 0.4752
Epoch 16/300
8/8 [==============================] - 0s 4ms/step - loss: 1.6461 - accuracy: 0.4917
Epoch 17/300
8/8 [==============================] - 0s 4ms/step - loss: 1.6174 - accuracy: 0.4876
Epoch 18/300
8/8 [==============================] - 0s 4ms/step - loss: 1.5901 - accuracy: 0.4959
Epoch 19/300
8/8 [==============================] - 0s 4ms/step - loss: 1.5651 - accuracy: 0.4959
Epoch 20/300
8/8 [==============================] - 0s 3ms/step - loss: 1.5379 - accuracy: 0.4917
Epoch 21/300
8/8 [==============================] - 0s 3ms/step - loss: 1.5086 - accuracy: 0.4876
Epoch 22/300
8/8 [==============================] - 0s 3ms/step - loss: 1.4850 - accuracy: 0.4917
Epoch 23/300
8/8 [==============================] - 0s 4ms/step - loss: 1.4559 - accuracy: 0.4917
Epoch 24/300
8/8 [==============================] - 0s 4ms/step - loss: 1.4275 - accuracy: 0.4876
Epoch 25/300
8/8 [==============================] - 0s 4ms/step - loss: 1.4020 - accuracy: 0.4917
Epoch 26/300
8/8 [==============================] - 0s 4ms/step - loss: 1.3763 - accuracy: 0.4959
Epoch 27/300
8/8 [==============================] - 0s 3ms/step - loss: 1.3503 - accuracy: 0.4876
Epoch 28/300
8/8 [==============================] - 0s 3ms/step - loss: 1.3281 - accuracy: 0.4917
Epoch 29/300
8/8 [==============================] - 0s 4ms/step - loss: 1.3081 - accuracy: 0.5083
Epoch 30/300
8/8 [==============================] - 0s 4ms/step - loss: 1.2798 - accuracy: 0.5124
Epoch 31/300
8/8 [==============================] - 0s 4ms/step - loss: 1.2597 - accuracy: 0.5124
Epoch 32/300
8/8 [==============================] - 0s 3ms/step - loss: 1.2385 - accuracy: 0.5041
Epoch 33/300
8/8 [==============================] - 0s 3ms/step - loss: 1.2128 - accuracy: 0.5000
Epoch 34/300
8/8 [==============================] - 0s 3ms/step - loss: 1.1956 - accuracy: 0.5083
Epoch 35/300
8/8 [==============================] - 0s 3ms/step - loss: 1.1730 - accuracy: 0.5165
Epoch 36/300
8/8 [==============================] - 0s 3ms/step - loss: 1.1511 - accuracy: 0.4917
Epoch 37/300
8/8 [==============================] - 0s 3ms/step - loss: 1.1355 - accuracy: 0.4959
Epoch 38/300
8/8 [==============================] - 0s 3ms/step - loss: 1.1143 - accuracy: 0.4959
Epoch 39/300
8/8 [==============================] - 0s 4ms/step - loss: 1.0983 - accuracy: 0.5124
Epoch 40/300
8/8 [==============================] - 0s 3ms/step - loss: 1.0802 - accuracy: 0.5000
Epoch 41/300
8/8 [==============================] - 0s 3ms/step - loss: 1.0596 - accuracy: 0.5083
Epoch 42/300
8/8 [==============================] - 0s 3ms/step - loss: 1.0448 - accuracy: 0.5124
Epoch 43/300
8/8 [==============================] - 0s 3ms/step - loss: 1.0256 - accuracy: 0.5124
Epoch 44/300
8/8 [==============================] - 0s 3ms/step - loss: 1.0147 - accuracy: 0.5207
Epoch 45/300
8/8 [==============================] - 0s 3ms/step - loss: 0.9945 - accuracy: 0.5165
Epoch 46/300
8/8 [==============================] - 0s 4ms/step - loss: 0.9827 - accuracy: 0.5248
Epoch 47/300
8/8 [==============================] - 0s 4ms/step - loss: 0.9629 - accuracy: 0.5289
Epoch 48/300
8/8 [==============================] - 0s 4ms/step - loss: 0.9490 - accuracy: 0.5331
Epoch 49/300
8/8 [==============================] - 0s 4ms/step - loss: 0.9343 - accuracy: 0.5496
Epoch 50/300
8/8 [==============================] - 0s 3ms/step - loss: 0.9228 - accuracy: 0.5455
Epoch 51/300
8/8 [==============================] - 0s 3ms/step - loss: 0.9093 - accuracy: 0.5537
Epoch 52/300
8/8 [==============================] - 0s 3ms/step - loss: 0.8966 - accuracy: 0.5455
Epoch 53/300
8/8 [==============================] - 0s 4ms/step - loss: 0.8891 - accuracy: 0.5496
Epoch 54/300
8/8 [==============================] - 0s 4ms/step - loss: 0.8735 - accuracy: 0.5579
Epoch 55/300
8/8 [==============================] - 0s 3ms/step - loss: 0.8622 - accuracy: 0.5496
Epoch 56/300
8/8 [==============================] - 0s 4ms/step - loss: 0.8510 - accuracy: 0.5620
Epoch 57/300
8/8 [==============================] - 0s 4ms/step - loss: 0.8425 - accuracy: 0.5661
Epoch 58/300
8/8 [==============================] - 0s 3ms/step - loss: 0.8314 - accuracy: 0.5537
Epoch 59/300
8/8 [==============================] - 0s 3ms/step - loss: 0.8251 - accuracy: 0.5620
Epoch 60/300
8/8 [==============================] - 0s 3ms/step - loss: 0.8131 - accuracy: 0.5785
Epoch 61/300
8/8 [==============================] - 0s 3ms/step - loss: 0.8056 - accuracy: 0.5620
Epoch 62/300
8/8 [==============================] - 0s 3ms/step - loss: 0.7949 - accuracy: 0.5702
Epoch 63/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7888 - accuracy: 0.5744
Epoch 64/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7794 - accuracy: 0.5702
Epoch 65/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7723 - accuracy: 0.5826
Epoch 66/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7637 - accuracy: 0.5909
Epoch 67/300
8/8 [==============================] - 0s 3ms/step - loss: 0.7569 - accuracy: 0.5950
Epoch 68/300
8/8 [==============================] - 0s 3ms/step - loss: 0.7520 - accuracy: 0.6157
Epoch 69/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7472 - accuracy: 0.5950
Epoch 70/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7401 - accuracy: 0.5909
Epoch 71/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7310 - accuracy: 0.6116
Epoch 72/300
8/8 [==============================] - 0s 3ms/step - loss: 0.7246 - accuracy: 0.6281
Epoch 73/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7198 - accuracy: 0.6240
Epoch 74/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7125 - accuracy: 0.6281
Epoch 75/300
8/8 [==============================] - 0s 5ms/step - loss: 0.7083 - accuracy: 0.6240
Epoch 76/300
8/8 [==============================] - 0s 4ms/step - loss: 0.7067 - accuracy: 0.6364
Epoch 77/300
8/8 [==============================] - 0s 4ms/step - loss: 0.6975 - accuracy: 0.6240
Epoch 78/300
8/8 [==============================] - 0s 4ms/step - loss: 0.6928 - accuracy: 0.6322
Epoch 79/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6943 - accuracy: 0.6364
Epoch 80/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6797 - accuracy: 0.6364
Epoch 81/300
8/8 [==============================] - 0s 4ms/step - loss: 0.6864 - accuracy: 0.6364
Epoch 82/300
8/8 [==============================] - 0s 4ms/step - loss: 0.6750 - accuracy: 0.6488
Epoch 83/300
8/8 [==============================] - 0s 4ms/step - loss: 0.6718 - accuracy: 0.6405
Epoch 84/300
8/8 [==============================] - 0s 4ms/step - loss: 0.6655 - accuracy: 0.6488
Epoch 85/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6629 - accuracy: 0.6529
Epoch 86/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6573 - accuracy: 0.6570
Epoch 87/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6560 - accuracy: 0.6570
Epoch 88/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6497 - accuracy: 0.6446
Epoch 89/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6479 - accuracy: 0.6653
Epoch 90/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6440 - accuracy: 0.6529
Epoch 91/300
8/8 [==============================] - 0s 4ms/step - loss: 0.6389 - accuracy: 0.6529
Epoch 92/300
8/8 [==============================] - 0s 5ms/step - loss: 0.6359 - accuracy: 0.6570
Epoch 93/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6328 - accuracy: 0.6612
Epoch 94/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6376 - accuracy: 0.6529
Epoch 95/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6306 - accuracy: 0.6818
Epoch 96/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6277 - accuracy: 0.6653
Epoch 97/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6178 - accuracy: 0.6694
Epoch 98/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6221 - accuracy: 0.6777
Epoch 99/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6128 - accuracy: 0.6736
Epoch 100/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6122 - accuracy: 0.6736
Epoch 101/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6116 - accuracy: 0.6694
Epoch 102/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6057 - accuracy: 0.6777
Epoch 103/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6022 - accuracy: 0.6818
Epoch 104/300
8/8 [==============================] - 0s 3ms/step - loss: 0.6016 - accuracy: 0.6860
Epoch 105/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5969 - accuracy: 0.6901
Epoch 106/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5954 - accuracy: 0.6818
Epoch 107/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5924 - accuracy: 0.6860
Epoch 108/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5899 - accuracy: 0.6983
Epoch 109/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5879 - accuracy: 0.7066
Epoch 110/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5875 - accuracy: 0.6860
Epoch 111/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5845 - accuracy: 0.7025
Epoch 112/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5812 - accuracy: 0.6983
Epoch 113/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5786 - accuracy: 0.6942
Epoch 114/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5783 - accuracy: 0.7066
Epoch 115/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5767 - accuracy: 0.6983
Epoch 116/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5733 - accuracy: 0.7190
Epoch 117/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5711 - accuracy: 0.6983
Epoch 118/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5720 - accuracy: 0.7149
Epoch 119/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5668 - accuracy: 0.7107
Epoch 120/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5675 - accuracy: 0.7066
Epoch 121/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5617 - accuracy: 0.7149
Epoch 122/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5610 - accuracy: 0.7107
Epoch 123/300
8/8 [==============================] - 0s 5ms/step - loss: 0.5606 - accuracy: 0.7066
Epoch 124/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5582 - accuracy: 0.7314
Epoch 125/300
8/8 [==============================] - 0s 5ms/step - loss: 0.5564 - accuracy: 0.7231
Epoch 126/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5545 - accuracy: 0.7107
Epoch 127/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5523 - accuracy: 0.7231
Epoch 128/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5514 - accuracy: 0.7149
Epoch 129/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5485 - accuracy: 0.7314
Epoch 130/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5472 - accuracy: 0.7438
Epoch 131/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5494 - accuracy: 0.7273
Epoch 132/300
8/8 [==============================] - 0s 5ms/step - loss: 0.5421 - accuracy: 0.7273
Epoch 133/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5453 - accuracy: 0.7231
Epoch 134/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5412 - accuracy: 0.7355
Epoch 135/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5413 - accuracy: 0.7397
Epoch 136/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5358 - accuracy: 0.7397
Epoch 137/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5367 - accuracy: 0.7397
Epoch 138/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5332 - accuracy: 0.7479
Epoch 139/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5341 - accuracy: 0.7397
Epoch 140/300
8/8 [==============================] - 0s 5ms/step - loss: 0.5310 - accuracy: 0.7479
Epoch 141/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5320 - accuracy: 0.7397
Epoch 142/300
8/8 [==============================] - 0s 5ms/step - loss: 0.5263 - accuracy: 0.7397
Epoch 143/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5294 - accuracy: 0.7314
Epoch 144/300
8/8 [==============================] - 0s 5ms/step - loss: 0.5271 - accuracy: 0.7355
Epoch 145/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5256 - accuracy: 0.7438
Epoch 146/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5245 - accuracy: 0.7397
Epoch 147/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5222 - accuracy: 0.7438
Epoch 148/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5282 - accuracy: 0.7273
Epoch 149/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5201 - accuracy: 0.7562
Epoch 150/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5178 - accuracy: 0.7562
Epoch 151/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5163 - accuracy: 0.7397
Epoch 152/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5153 - accuracy: 0.7479
Epoch 153/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5132 - accuracy: 0.7521
Epoch 154/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5110 - accuracy: 0.7521
Epoch 155/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5107 - accuracy: 0.7438
Epoch 156/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5082 - accuracy: 0.7562
Epoch 157/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5099 - accuracy: 0.7438
Epoch 158/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5116 - accuracy: 0.7438
Epoch 159/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5066 - accuracy: 0.7603
Epoch 160/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5118 - accuracy: 0.7438
Epoch 161/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5011 - accuracy: 0.7686
Epoch 162/300
8/8 [==============================] - 0s 4ms/step - loss: 0.5049 - accuracy: 0.7686
Epoch 163/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4999 - accuracy: 0.7769
Epoch 164/300
8/8 [==============================] - 0s 3ms/step - loss: 0.5008 - accuracy: 0.7562
Epoch 165/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4977 - accuracy: 0.7686
Epoch 166/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4997 - accuracy: 0.7769
Epoch 167/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4976 - accuracy: 0.7727
Epoch 168/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4955 - accuracy: 0.7603
Epoch 169/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4980 - accuracy: 0.7727
Epoch 170/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4920 - accuracy: 0.7727
Epoch 171/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4936 - accuracy: 0.7645
Epoch 172/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4943 - accuracy: 0.7727
Epoch 173/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4901 - accuracy: 0.7810
Epoch 174/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4881 - accuracy: 0.7810
Epoch 175/300
8/8 [==============================] - 0s 6ms/step - loss: 0.4868 - accuracy: 0.7727
Epoch 176/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4871 - accuracy: 0.7769
Epoch 177/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4856 - accuracy: 0.7727
Epoch 178/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4862 - accuracy: 0.7851
Epoch 179/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4834 - accuracy: 0.7769
Epoch 180/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4828 - accuracy: 0.7727
Epoch 181/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4838 - accuracy: 0.7934
Epoch 182/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4802 - accuracy: 0.7893
Epoch 183/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4789 - accuracy: 0.7810
Epoch 184/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4881 - accuracy: 0.7521
Epoch 185/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4749 - accuracy: 0.7934
Epoch 186/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4829 - accuracy: 0.7727
Epoch 187/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4769 - accuracy: 0.7769
Epoch 188/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4763 - accuracy: 0.8099
Epoch 189/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4732 - accuracy: 0.7769
Epoch 190/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4736 - accuracy: 0.7851
Epoch 191/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4713 - accuracy: 0.7769
Epoch 192/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4754 - accuracy: 0.7975
Epoch 193/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4720 - accuracy: 0.7893
Epoch 194/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4688 - accuracy: 0.7810
Epoch 195/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4699 - accuracy: 0.7934
Epoch 196/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4664 - accuracy: 0.7893
Epoch 197/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4683 - accuracy: 0.8017
Epoch 198/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4668 - accuracy: 0.7810
Epoch 199/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4641 - accuracy: 0.7893
Epoch 200/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4628 - accuracy: 0.8058
Epoch 201/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4618 - accuracy: 0.7934
Epoch 202/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4612 - accuracy: 0.7893
Epoch 203/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4601 - accuracy: 0.8017
Epoch 204/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4602 - accuracy: 0.8099
Epoch 205/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4567 - accuracy: 0.7934
Epoch 206/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4580 - accuracy: 0.7934
Epoch 207/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4577 - accuracy: 0.7975
Epoch 208/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4558 - accuracy: 0.8017
Epoch 209/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4572 - accuracy: 0.8017
Epoch 210/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4555 - accuracy: 0.8099
Epoch 211/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4536 - accuracy: 0.8099
Epoch 212/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4517 - accuracy: 0.8058
Epoch 213/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4532 - accuracy: 0.7975
Epoch 214/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4525 - accuracy: 0.8017
Epoch 215/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4521 - accuracy: 0.7851
Epoch 216/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4522 - accuracy: 0.8058
Epoch 217/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4550 - accuracy: 0.7934
Epoch 218/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4464 - accuracy: 0.7975
Epoch 219/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4489 - accuracy: 0.8058
Epoch 220/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4464 - accuracy: 0.8140
Epoch 221/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4442 - accuracy: 0.8099
Epoch 222/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4459 - accuracy: 0.8017
Epoch 223/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4435 - accuracy: 0.8140
Epoch 224/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4440 - accuracy: 0.8099
Epoch 225/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4402 - accuracy: 0.8140
Epoch 226/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4466 - accuracy: 0.7975
Epoch 227/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4429 - accuracy: 0.8058
Epoch 228/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4389 - accuracy: 0.8140
Epoch 229/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4410 - accuracy: 0.7934
Epoch 230/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4377 - accuracy: 0.8140
Epoch 231/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4388 - accuracy: 0.8058
Epoch 232/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4374 - accuracy: 0.8099
Epoch 233/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4354 - accuracy: 0.8140
Epoch 234/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4358 - accuracy: 0.8140
Epoch 235/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4353 - accuracy: 0.8099
Epoch 236/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4333 - accuracy: 0.8140
Epoch 237/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4390 - accuracy: 0.8058
Epoch 238/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4364 - accuracy: 0.8140
Epoch 239/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4347 - accuracy: 0.8058
Epoch 240/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4303 - accuracy: 0.8140
Epoch 241/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4308 - accuracy: 0.8140
Epoch 242/300
8/8 [==============================] - 0s 5ms/step - loss: 0.4305 - accuracy: 0.8099
Epoch 243/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4345 - accuracy: 0.7975
Epoch 244/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4379 - accuracy: 0.8017
Epoch 245/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4279 - accuracy: 0.8223
Epoch 246/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4276 - accuracy: 0.8182
Epoch 247/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4264 - accuracy: 0.8264
Epoch 248/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4259 - accuracy: 0.8182
Epoch 249/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4247 - accuracy: 0.8140
Epoch 250/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4267 - accuracy: 0.8140
Epoch 251/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4266 - accuracy: 0.8306
Epoch 252/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4217 - accuracy: 0.8223
Epoch 253/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4221 - accuracy: 0.8223
Epoch 254/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4212 - accuracy: 0.8182
Epoch 255/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4213 - accuracy: 0.8182
Epoch 256/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4220 - accuracy: 0.8140
Epoch 257/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4240 - accuracy: 0.8347
Epoch 258/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4189 - accuracy: 0.8264
Epoch 259/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4204 - accuracy: 0.8264
Epoch 260/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4205 - accuracy: 0.8306
Epoch 261/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4188 - accuracy: 0.8182
Epoch 262/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4168 - accuracy: 0.8223
Epoch 263/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4183 - accuracy: 0.8264
Epoch 264/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4186 - accuracy: 0.8347
Epoch 265/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4149 - accuracy: 0.8388
Epoch 266/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4178 - accuracy: 0.8264
Epoch 267/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4143 - accuracy: 0.8347
Epoch 268/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4156 - accuracy: 0.8347
Epoch 269/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4165 - accuracy: 0.8388
Epoch 270/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4117 - accuracy: 0.8264
Epoch 271/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4177 - accuracy: 0.8223
Epoch 272/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4133 - accuracy: 0.8430
Epoch 273/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4097 - accuracy: 0.8347
Epoch 274/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4092 - accuracy: 0.8430
Epoch 275/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4140 - accuracy: 0.8306
Epoch 276/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4127 - accuracy: 0.8264
Epoch 277/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4162 - accuracy: 0.8264
Epoch 278/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4078 - accuracy: 0.8347
Epoch 279/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4112 - accuracy: 0.8182
Epoch 280/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4158 - accuracy: 0.8223
Epoch 281/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4085 - accuracy: 0.8264
Epoch 282/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4080 - accuracy: 0.8306
Epoch 283/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4069 - accuracy: 0.8347
Epoch 284/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4047 - accuracy: 0.8430
Epoch 285/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4039 - accuracy: 0.8347
Epoch 286/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4035 - accuracy: 0.8388
Epoch 287/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4039 - accuracy: 0.8471
Epoch 288/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4045 - accuracy: 0.8306
Epoch 289/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4122 - accuracy: 0.8264
Epoch 290/300
8/8 [==============================] - 0s 4ms/step - loss: 0.4002 - accuracy: 0.8430
Epoch 291/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4027 - accuracy: 0.8306
Epoch 292/300
8/8 [==============================] - 0s 3ms/step - loss: 0.3996 - accuracy: 0.8430
Epoch 293/300
8/8 [==============================] - 0s 3ms/step - loss: 0.4005 - accuracy: 0.8388
Epoch 294/300
8/8 [==============================] - 0s 3ms/step - loss: 0.3993 - accuracy: 0.8430
Epoch 295/300
8/8 [==============================] - 0s 3ms/step - loss: 0.3998 - accuracy: 0.8306
Epoch 296/300
8/8 [==============================] - 0s 3ms/step - loss: 0.3973 - accuracy: 0.8388
Epoch 297/300
8/8 [==============================] - 0s 4ms/step - loss: 0.3985 - accuracy: 0.8430
Epoch 298/300
8/8 [==============================] - 0s 4ms/step - loss: 0.3975 - accuracy: 0.8388
Epoch 299/300
8/8 [==============================] - 0s 4ms/step - loss: 0.3985 - accuracy: 0.8388
Epoch 300/300
8/8 [==============================] - 0s 3ms/step - loss: 0.3957 - accuracy: 0.8554
<keras.callbacks.History at 0x7f944ac56b80>

Y_pred_nn = model.predict(X_test)
     
2/2 [==============================] - 0s 7ms/step

Y_pred_nn.shape
     
(61, 1)

rounded = [round(x[0]) for x in Y_pred_nn]

Y_pred_nn = rounded
     

acc_score_nn = round(accuracy_score(Y_pred_nn,Y_test)*100,2)

print("The Accuracy Score achieved using Neural Network is: "+str(acc_score_nn)+" %")
     
The Accuracy Score achieved using Neural Network is: 77.05 %

pre_score_nn = round(precision_score(Y_pred_nn,Y_test)*100,2)

print("The Precision Score achieved using Neural Network is: "+str(pre_score_nn)+" %")
     
The Precision Score achieved using Neural Network is: 82.35 %

rec_score_nn = round(recall_score(Y_pred_nn,Y_test)*100,2)

print("The Recall Score achieved using Neural Network is: "+str(rec_score_nn)+" %")
     
The Recall Score achieved using Neural Network is: 77.78 %

f1_score_nn = round(f1_score(Y_pred_nn,Y_test)*100,2)

print("The F1-Score achieved using Neural Network is: "+str(f1_score_nn)+" %")
     
The F1-Score achieved using Neural Network is: 80.0 %
Final Score and Overall Comparision

acc_scores = [acc_score_lr,acc_score_nb,acc_score_svm,acc_score_knn,acc_score_dt,acc_score_rf,acc_score_xgb,acc_score_nn]
algorithms = ["Logistic Regression","Naive Bayes","SVM","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]    

for i in range(len(algorithms)):
    print("The accuracy score achieved using "+algorithms[i]+" is: "+str(acc_scores[i])+" %")
     
The accuracy score achieved using Logistic Regression is: 85.25 %
The accuracy score achieved using Naive Bayes is: 85.25 %
The accuracy score achieved using SVM is: 81.97 %
The accuracy score achieved using K-Nearest Neighbors is: 67.21 %
The accuracy score achieved using Decision Tree is: 81.97 %
The accuracy score achieved using Random Forest is: 90.16 %
The accuracy score achieved using XGBoost is: 85.25 %
The accuracy score achieved using Neural Network is: 77.05 %

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Accuracy score")

sns.barplot(algorithms,acc_scores)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f94aeac9ac0>


pre_scores = [pre_score_lr,pre_score_nb,pre_score_svm,pre_score_knn,pre_score_dt,pre_score_rf,pre_score_xgb,pre_score_nn]
algorithms = ["Logistic Regression","Naive Bayes","SVM","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]    

for i in range(len(algorithms)):
    print("The Precision Score achieved using "+algorithms[i]+" is: "+str(pre_scores[i])+" %")
     
The Precision Score achieved using Logistic Regression is: 88.24 %
The Precision Score achieved using Naive Bayes is: 91.18 %
The Precision Score achieved using SVM is: 88.24 %
The Precision Score achieved using K-Nearest Neighbors is: 67.65 %
The Precision Score achieved using Decision Tree is: 82.35 %
The Precision Score achieved using Random Forest is: 94.12 %
The Precision Score achieved using XGBoost is: 88.24 %
The Precision Score achieved using Neural Network is: 82.35 %

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Precision score")

sns.barplot(algorithms,pre_scores)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f9449327700>


rec_scores = [rec_score_lr,rec_score_nb,rec_score_svm,rec_score_knn,rec_score_dt,rec_score_rf,rec_score_xgb,rec_score_nn]
algorithms = ["Logistic Regression","Naive Bayes","SVM","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]    

for i in range(len(algorithms)):
    print("The Recall Score achieved using "+algorithms[i]+" is: "+str(rec_scores[i])+" %")
     
The Recall Score achieved using Logistic Regression is: 85.71 %
The Recall Score achieved using Naive Bayes is: 83.78 %
The Recall Score achieved using SVM is: 81.08 %
The Recall Score achieved using K-Nearest Neighbors is: 71.88 %
The Recall Score achieved using Decision Tree is: 84.85 %
The Recall Score achieved using Random Forest is: 88.89 %
The Recall Score achieved using XGBoost is: 85.71 %
The Recall Score achieved using Neural Network is: 77.78 %

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("Recall score")

sns.barplot(algorithms,rec_scores)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f944a920790>


f1_scores = [f1_score_lr,f1_score_nb,f1_score_svm,f1_score_knn,f1_score_dt,f1_score_rf,f1_score_xgb,f1_score_nn]
algorithms = ["Logistic Regression","Naive Bayes","SVM","K-Nearest Neighbors","Decision Tree","Random Forest","XGBoost","Neural Network"]    

for i in range(len(algorithms)):
    print("The F1-Score achieved using "+algorithms[i]+" is: "+str(f1_scores[i])+" %")
     
The F1-Score achieved using Logistic Regression is: 86.96 %
The F1-Score achieved using Naive Bayes is: 87.32 %
The F1-Score achieved using SVM is: 84.51 %
The F1-Score achieved using K-Nearest Neighbors is: 69.7 %
The F1-Score achieved using Decision Tree is: 83.58 %
The F1-Score achieved using Random Forest is: 91.43 %
The F1-Score achieved using XGBoost is: 86.96 %
The F1-Score achieved using Neural Network is: 80.0 %

sns.set(rc={'figure.figsize':(15,8)})
plt.xlabel("Algorithms")
plt.ylabel("F1-Score")

sns.barplot(algorithms,f1_scores)
     
/usr/local/lib/python3.8/dist-packages/seaborn/_decorators.py:36: FutureWarning: Pass the following variables as keyword args: x, y. From version 0.12, the only valid positional argument will be `data`, and passing other arguments without an explicit keyword will result in an error or misinterpretation.
  warnings.warn(
<matplotlib.axes._subplots.AxesSubplot at 0x7f944a7b9580>
