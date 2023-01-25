# Calories_Burnt_Prediction-

This document is written to provide aid in understanding the project.

Contents of the document - 
Understanding the problem statement 
About the dataset
Machine learning 
Types of machine learning models with examples 
Machine learning algorithm used for the model - XGBoost Regression model 
NumPy library 
Pandas library 
Scikit-learn library 
Matplot library 
Exploratory data analysis 
Fixing missing values in the dataset 
Data visualisation - Seaborn 
Correlation and heatmap 
Categorical encoding 
Train-test split 
Model evaluation - Mean Absolute Error 

What is the problem statement for the machine learning algorithm ?
What are the metabolic activities in a human body when a person engages in any sort of physical activity. This will help us understand the problem better and create a good hypothesis and work flow statement. Let's consider a person is exercising. For example, they are on a treadmill, or running or yoga or any other form of exercise. They have consumed a good meal. So energy derived from this meal (carbohydrates broken down into simple sugar such as glucose which is further broken down in the presence of oxygen into energy for the body) will be used to perform the exercise properly. The muscles require a great amount of energy. This is supplied through the blood. To meet the high oxygen demand there is greater supply of oxygen. In turn, we observe that there is an increase in the heartbeat. Heartbeat may be more than 100 bpm. There is also an increase in body temperature. Because of this increase temperature, the body releases sweat (as a counteraction to deal with the high body temperature). This helps in burning calories. 

Therefore, we will use body temperature and sweat as a parameter to calculate the amount of calories burnt.

About the dataset - 
The dataset contains the following columns - 
User_ID - identification number of the user 
Calories - amount of calories burnt after the exercise 
Gender 
Age
Height 
Weight 
Duration - total duration of exercise 
Heart_Rate - speed at which the heart beats. average is 72 beats per minute
Body_Temp - body temperature after exercise
It consists of 15000 rows (15000 data points) and 9 columns (9 features).

Machine learning - 
Machine learning enables the processing of sonar signals and target detection. Machine Learning is a subset of Artificial Intelligence. This involves the development of computer systems that are able to learn by using algorithms and statistical measures to study data and draw results from it. Machine learning is basically an integration of computer systems, statistical mathematics and data.

Machine Learning is further divided into three classes - Supervised learning, Unsupervised learning and Reinforcement Learning. 

Supervised learning is a machine learning method in which models are trained using labelled data. In supervised learning, models need to find the mapping function and find a relationship between the input and output. In this, the user has a somewhat idea of what the output should look like. It is of two types - regression (predicts results with continuous output. For example, given the picture of a person, we have to predict their age on the basis of the given picture) and classification (predict results in a discrete output. For example, given a patient with a tumor, we have to predict whether the tumor is malignant or benign.) 

Unsupervised learning is a method in which patterns are inferred from the unlabelled input data. It allows us to approach problems with little or no idea what the results should look like. We can derive structure from the data where we don’t necessarily know the effect of variables. We can derive the structure by clustering the data based on relationships among the variables in the data. With unsupervised learning there is no feedback on the prediction results. It is of two types - clustering (model groups input data into groups that are somehow similar or related by different variables. For example, clustering data of thousands of genes into groups) and non-clustering (models identifies individual inputs. It helps us find structure in a chaotic environment. For example, the cocktail party problem where we need to identify different speakers from a given audiotape.)

Reinforcement learning is a feedback-based machine learning technique. It is about taking suitable action to maximise reward in a particular situation. For example, a robotic dog learning the movement of his arms or teaching self-driving cars how to depict the best route for travelling. 

For this project, I will use XGBoost Regressor 

Extreme Gradient Boost 

XGBoost is a powerful approach for building supervised regression models. It is an efficient and effective implementation of the gradient boosting algorithm. The objective function contains loss function and a regularization term. It tells about the difference between actual values and predicted values, i.e how far the model results are from the real values.

Libraries used in the project - 
NumPy  
It is a python library used for working with arrays. It has functions for working in the domain of linear algebra, fourier transform, and matrices. It is the fundamental package for scientific computing with python. NumPy stands for numerical python. 

NumPy is preferred because it is faster than traditional python lists. It has supporting functions that make working with ndarray very easy. Arrays are frequently used where speed and resources are very important. NumPy arrays are faster because it is stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently. This is locality of reference in computer science. 

Pandas - 
Pandas is made for working with relational or labelled data both easily and intuitively. It provides various data structures and operations for manipulating numerical data and time series. 

It has a lot of advantages like - 
Fast and efficient for manipulating and analyzing data
Data from different file objects can be loaded 
Easy handling of missing data in data preprocessing 
Size mutability 
Easy dataset merging and joining 
Flexible reshaping and pivoting of datasets 
Gives time-series functionality 

Pandas is built on top of NumPy library. That means that a lot of structures of NumPy are used or replicated in Pandas. The data produced by pandas are often used as input for plotting functions of Matplotlib, statistical analysis in SciPy, and machine learning algorithms in Scikit-learn. 

Scikit-Learn - 
It provides efficient tools for machine learning and statistical modeling including classification, regression, clustering and dimensionality reduction. It has numerous machine learning, pre-processing, cross validation, and visualization algorithms. 

Maplotlib - 
Matplotlib is a visualization library in python for 2D plots of arrays. It allows visual access to huge amounts of data in easily digestible visuals and plots like line, bar, scatter, histogram etc. 

Exploratory data analysis - 
Exploratory data analysis is the process of performing initial investigation on the data to discover patterns or spot anomalies. It is done to test the hypothesis and to check assumptions with the help of summary statistics and graphical representations. 

‘describe()’ method returns description of data in DataFrame. It tells us the following information for each column - 
Count - number of non-empty values
Mean - the average (mean) value  
Std - standard deviation
Min - minimum value
25% - the 25 percentile 
50% - the 50 percentile 
75% - the 75 percentile
Max - maximum value

The info() method prints the information about dataframe. 
It contains the number of columns, column labels, column data types, memory usage, range index, and number of cells in each column. 

Parameters - 
verbose - It is used to print the full summary of the dataset.
buf - It is a writable buffer, default to sys.stdout.
max_cols - It specifies whether a half summary or full summary is to be printed.
memory_usage - It specifies whether total memory usage of the DatFrame elements (including index) should be displayed.
null_counts - It is used to show the non-null counts.

Missing values - 
Missing values are common when working with real-world datasets. Missing data could result from a human factor, a problem in electrical sensors, missing files, improper management or other factors. Missing values can result in loss of significant information. Missing value can bias the results of model and reduce the accuracy of the model. There are various methods of handling missing data but unfortunately they still introduce some bias such as favoring one class over the other but these methods are useful. 

In Pandas, missing values are represented by NaN. It stands for Not a Number. 

Reasons for missing values - 
Past data may be corrupted due to improper maintenance
Observations are not recorded for certain fields due to faulty measuring equipments. There might by a failure in recording the values due to human error. 
The user has not provided the values intentionally. 

Why we need to handle missing values - 
Many machine learning algorithms fail if the dataset contains missing values. 
Missing values may result in a biased machine learning model which will lead to incorrect results if the missing values are not handled properly. 
Missing data can lead to lack of precision. 

Types of missing data - 
Understanding the different types of missing data will provide insights into how to approach the missing values in the dataset. 
Missing Completely at Random (MCAR) 
There is no relationship between the missing data and any other values observed or unobserved within the given dataset. Missing values are completely independent of other data. There is no pattern. The probability of data being missing is the same for all the observations. 
The data may be missing due to human error, some system or equipment failure, loss of sample, or some unsatisfactory technicalities while recording the values.
It should not be assumed as it’s a rare case. The advantage of data with such missing values is that the statistical analysis remains unbiased.   
Missing at Random (MAR)
The reason for missing values can be explained by variables on which complete information is provided. There is relationship between the missing data and other values/data. In this case, most of the time, data is not missing for all the observations. It is missing only within sub-samples of the data and there is pattern in missing values. 
In this, the statistical analysis might result in bias. 
Not MIssing at Random (NMAR)
Missing values depend on unobserved data. If there is some pattern in missing data and other observed data can not explain it. If the missing data does not fall under the MCAR or MAR then it can be categorized as MNAR. 
It can happen due to the reluctance of people in providing the required information. 
In this case too, statistical analysis might result in bias. 

How to handle missing values - 

isnull().sum() - shows the total number of missing values in each columns 

We need to analyze each column very carefully to understand the reason behind missing values. There are two ways of handling values - 
Deleting missing values - this is a simple method. If the missing value belongs to MAR and MCAR then it can be deleted. But if the missing value belongs to MNAR then it should not be deleted. 
The disadvantage of this method is that we might end up deleting useful data. 
You can drop an entire column or an entire row. 
Imputing missing values - there are various methods of imputing missing values
Replacing with arbitrary value 
Replacing with mean - most common method. But in case of outliers, mean will not be appropriate
Replacing with mode - mode is most frequently occuring value. It is used in case of categorical features. 
Replacing with median - median is middlemost value. It is better to use median in case of outliers. 
Replacing with previous value - it is also called a forward fill. Mostly used in time series data. 
Replacing with next value - also called backward fill. 
Interpolation 

Data visualisation - 
Datasets often come in csv files, spreadsheets, table form etc. Data visualisation provides a good and organized pictorial representation of data which makes it easier to observe, understand and analyze. 
Python provides various libraries that come with different features for visualizing data. All these libraries have different features and can support various types of graphs. 
Matplotlib - for 2D array plots. It includes wide range of plots, such as scatter, line, bar, histogram and others that can assist in delving deeper into trends. 
Seaborn - it is used for creating statistical representations based on datasets. It is built on top of matplotlib. It is built on top of pandas’ data structures. The library conducts the necessary modelling and aggregation internally to create insightful visuals.
Bokeh - it is a modern web browser based interactive visualization library. It can create engaging plots and dashboards with huge streaming data. The library contains many intuitive graphs. It has close relationship with PyData tools. The library is ideal for creating customized visuals.  
Plotly - python visualization library that is interactive, accessible, high-level and browser-based. Scientific graphs, 3D charts, statistical plots, and financial charts. Interaction and editing options are available. 

Data visualisation plot used in the project - 
Seaborn - 
Seaborn aims to make visualization the central part of exploring and understanding data. It provides dataset-oriented APIs, so that we can switch between different visual representations for same variables for better understanding of dataset.

Distplot - 
It is used for univariant set of observations and visualizes it through a histogram - only one observation and hence we choose one particular column of the dataset. 

Countplot - 
seaborn.countplot() method is used to show the counts of observations in each categorical bin using bars. 

Parameters - 
x, y - This parameter take names of variables in data or vector data, optional, Inputs for plotting long-form data.
Hue - (optional) This parameter take column name for colour encoding.
Data - (optional) This parameter take DataFrame, array, or list of arrays, Dataset for plotting. If x and y are absent, this is interpreted as wide-form. Otherwise it is expected to be long-form.
order, hue_order - (optional) This parameter take lists of strings. Order to plot the categorical levels in, otherwise the levels are inferred from the data objects.
Orient - (optional)This parameter take “v” | “h”, Orientation of the plot (vertical or horizontal). This is usually inferred from the dtype of the input variables but can be used to specify when the “categorical” variable is a numeric or when plotting wide-form data.
Color - (optional) This parameter take matplotlib color, Color for all of the elements, or seed for a gradient palette.
Palette - (optional) This parameter take palette name, list, or dict, Colors to use for the different levels of the hue variable. Should be something that can be interpreted by color_palette(), or a dictionary mapping hue levels to matplotlib colors.
Saturation - (optional) This parameter take float value, Proportion of the original saturation to draw colors at. Large patches often look better with slightly desaturated colors, but set this to 1 if you want the plot colors to perfectly match the input color spec.
Dodge - (optional) This parameter take bool value, When hue nesting is used, whether elements should be shifted along the categorical axis.
Ax - (optional) This parameter take matplotlib Axes, Axes object to draw the plot onto, otherwise uses the current Axes.
Kwargs - This parameter take key, value mappings, Other keyword arguments are passed through to matplotlib.axes.Axes.bar().
Returns - Returns the Axes object with the plot drawn onto it.

corr() - 
corr() method finds the pairwise correlation of each column in a dataframe. 

There are two types of correlation - 
Positive correlation - when the columns are directly proportional to each other 
Negative correlation - when the columns are inversely proportional to each other. 

Heatmap - 
Heatmap is graphical representation of data using colors to visualize value of the matrix. It is a plot of rectangular data as color-encoded matrix. 

Parameters:
data: 2D dataset that can be coerced into an ndarray.
vmin, vmax: Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
cmap: The mapping from data values to color space.
center: The value at which to center the colormap when plotting divergent data.
annot: If True, write the data value in each cell.
fmt: String formatting code to use when adding annotations.
linewidths: Width of the lines that will divide each cell.
linecolor: Color of the lines that will divide each cell.
cbar: Whether to draw a colorbar.


Categorical data encoding - 
Datasets contain multiple labels (in the form of words of numbers) in one or more than one columns. Training data is often labelled in words to make the data more human-readable and understandable. Categorical data encoding is coding data to convert it to a machine-readable form. It is an important data preprocessing method. 

There are two types of categorical data 
Ordinal data - categories have an inherent order
Nominal data - no inherent order of categories

Label encoding - 
Label encoding refers to converting labels into numeric form to convert it to machine-readable form. It is an important data preprocessing method for structured dataset in supervised learning. 
Limitation - label encoding assigns a unique number to each class of data. However, this can lead to generation of priority issues in the training of data. A label with high value may be considered to have high priority than a label having a lower value.  

replace() - 
replace() in Python returns a copy of the string where all occurrences of a substring are replaced with another substring.

Parameters - 
Old – old substring you want to replace.
New – new substring which would replace the old substring.
Count – (Optional ) the number of times you want to replace the old substring with the new substring. 

Return value - It returns a copy of the string where all occurrences of a substring are replaced with another substring.

Train-test split - 
The entire dataset is split into training dataset and testing dataset. Usually, 80-20 or 70-30 split is done. The train-test split is used to prevent the model from overfitting and to estimate the performance of prediction-based algorithms. We need to split the dataset to evaluate how well our machine learning model performs. The train set is used to fit the model, and statistics of training set are known. Test set is for predictions. 

This is done by using scikit-learn library and train_test_split() function. 
Parameters - 
*arrays: inputs such as lists, arrays, data frames, or matrices
test_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our test size. its default value is none.
train_size: this is a float value whose value ranges between 0.0 and 1.0. it represents the proportion of our train size. its default value is none.
random_state: this parameter is used to control the shuffling applied to the data before applying the split. it acts as a seed.
shuffle: This parameter is used to shuffle the data before splitting. Its default value is true.
stratify: This parameter is used to split the data in a stratified fashion.

Model evaluation - 

Model evaluation is done to test the performance of machine learning model. It is done to determine whether the model is a good fit for the input dataset or not. 

For this project, I will use Mean Absolute Error.

It calculates the average difference between the calculated values and the actual values. It is used as evaluation metrics for regression models for predicting the accuracy of the machine learning model. 

The closer the value of MAE is to 0, the better the model. 
