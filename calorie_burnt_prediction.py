#Calorie burnt prediction 

#First of all it's important to understand more about the problem statment. 

#What are the metabolic activities in a human body when a person engages in any sort of physical activity. This will help 
#us understand the problem better and create a good hypothesis and work flow statement. 
#Let's consider a person is exercising. For example, they are on a treadmill, or running or yoga or any other form of 
#exercise. They have consumed a good meal. So energy derived from this meal (carbohydrates broken down into simple sugar 
#such as glucose which is further broken down in the presence of oxygen into energy for the body) will be used to perform 
#the exercise properly. The muscles require a great amount of energy. This is supplied through the blood. To meet the high 
#oxygen demand there is greater supply of oxygen. In turn, we observe that there is an increase in the heartbeat. Heartbeat 
#may be more than 100 bpm. There is also an increase in body temperature. Because of this increase temperature, the body 
#releases sweat (as a counteraction to deal with the high body temperature). This helps in burning calories. 

#Therefore, we will use body temperature and sweat as a parameter to calculate the amount of calories burnt. 

#workflow 
#1. collect the data 
#2. data preprocessing
#3. data analysis 
#4. train test split
#5. model used - XGBoostRegressor
#6. model evaluation - mean absolute error 

#import the libraries 
#linear algebra - for building matrices 
import numpy as np 

#data preprocessing and exploration
import pandas as pd 

#data visualisation 
import matplotlib.pyplot as plt 
import seaborn as sns

#model building and evaluation 
from sklearn.model_selection import train_test_split 
from xgboost import XGBRegressor
from sklearn import metrics

#data collection and processing 
#loading data from csv file to pandas dataframe 

#calories data 
calories = pd.read_csv('calories_data.csv')

#view the first five rows of the data 
calories.head()
#it contains two columns - 
#1. User_ID - identification number of the user 
#2. Calories - amount of calories burnt after the exercise 

#exercise data 
exercise_data = pd.read_csv("exercise_data.csv")

#view the first five rows of the data 
exercise_data.head()
#dataset contains the following columns - 
#1. User_ID
#2. Gender 
#3. Age
#4. Height 
#5. Weight 
#6. Duration - total duration of exercise 
#7. Heart_Rate - speed at which the heart beats. average is 72 beats per minute
#8. Body_Temp - body temperature after exercise

#combine the two dataframes 
calories_data = pd.concat([exercise_data, calories['Calories']], axis=1)

#view the first five rows of the combined dataframe
calories_data.head()

#view the total number of rows and columns 
calories_data.shape
#there are 15000 rows (15000 data points) and 9 columns (9 features)

#more information about the dataframe
calories_data.info()

#check for missing values 
calories_data.isnull().sum()
#we see that there are no null points 

#data analysis 
#statistical measures of the data 
calories_data.describe()

#data visualisation 
sns.set()

#let's see the distribution of gender. plotting gender column in countplot
sns.countplot(calories_data['Gender'])

#finding the distribution of age column 
sns.displot(calories_data['Age'])

#finding the distribution of height column 
sns.displot(calories_data['Height'])

#finding the distribution of weight column 
sns.displot(calories_data['Weight'])

#finding the distribution of distribution column 
sns.displot(calories_data['Distribution'])

#finding the distribution of heart rate column 
sns.displot(calories_data['Heart_Rate'])

#finding correlation in dataset 
correlation = calories_data.corr()
#there are two types of correlations - positive correlation and negative correlation. 
#positive correlation means that two columns are directly proportional and negative correlation means that two columns 
#are inversely proportional. 

#construct heatmap to understand the correlation 
plt.figure(figsize=(10,10))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')

#convert text to numerical values 
calories_data.replace({'Gender':{'male':0, 'female':1}}, inplace=True)

#separate features and target 
X = calories_data.drop(columns = ['Calories', 'User_ID'], axis=1)
Y = calories_data['Calories']

#train test split 
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=2)

#train model 
model = XGBRegressor() 
model.fit(X_train, Y_train)

#model evaluation 
#prediction on test data 
test_data_prediction = model.predict(X_test)

#mean absolute error 
mae = metrics.mean_absolute_error(Y_test, test_data_prediction)
print ('mean absolute error', mae)
#the mean absolute error is 2.71