# Water quality predictor 


## Introduction

We all love recreation activities in and around water. We always wonder where we should go, and which locations have the safest water for
recreational activities, as well as which locations have potable water.

In this project we will develop a tool which will show lakes around King County, Washington State, and display which are safe for
recreational and potability use. It is an interactive website which takes user selection of a specific lake and presents the water 
quality forecast for that lake. The location will be displayed using marker layer map and pop-up markers, a detailed water quality prediction
table as well as a water quality prediction graph.


## Database Roadmap

- Create QuickDBD 

![QuickDBD-export](https://user-images.githubusercontent.com/106849689/200139417-deccad9a-fb82-4ff2-92a1-862dc4d4b325.png)

- Use Big Lake Sammamish data to create Database for Lake Sammamish

![Tables_in_db](https://user-images.githubusercontent.com/106849689/204070754-67ab3d42-1636-4fb2-bd6b-0a43a0978aa9.png)

- Use SQLite for Database 

- Make a connection to the SQL Database

<img width="1269" alt="save_to_db" src="https://user-images.githubusercontent.com/106849689/204071145-a11c75bd-0dcd-4a1a-8cd0-a9e32a0a060a.png">

- Create final tables for Database 

![King_County_WQI_final_Tables](https://user-images.githubusercontent.com/106849689/204071310-b9945dd5-1b86-4e14-9a5f-43eaa8e9a93d.png)

- Create an Engine to Database 

- Inspect Table / Commit to Database 

<img width="1269" alt="commit_db" src="https://user-images.githubusercontent.com/106849689/204071694-c926fce0-c0ef-4633-93b2-01e594fba05a.png">

## Machine Learning Models for Predicting Water quality Index of Lakes in Washington state

## Water Quality Index:

The water quality index (WQI) is extensively used to assess and classify the quality of surface water and groundwater. The water quality index is computed based on the physicochemical parameters of the water such as, temperature, pH, turbidity, dissolved oxygen (DO), biochemical oxygen demand (BOD), and concentrations of other pollutants), for the estimation of water quality.  WQI provides a meaningful way to categorize the quality of any water resource in some quantitative form, which could help decision makers and planners make well informed decisions on the subject of water resources management. However, it involves lengthy calculations to formulate the water quality index and hence, requires a lot of resources in terms of time and effort. To solve this problem an alternative approach is needed, which is more efficient and accurate to estimate the WQI.

Machine learning (ML) has proven to be a cutting edge tool for modeling complex non-linear behaviors in water resources research and has also been 
used for assessing the water quality. However, there are many techniques in ML which could be applied to estimate and predict the WQI. In the first
section of the project we would like to explore the different techniques of machine learning and compare them with each other in terms of their
accuracy. We would like to choose the one technique which is most suitable for the kind of data we have in order to predict the WQI of lakes in
Washington state. We are going to refer the article Khoi et al., (2022) for choosing the approach to perform the ML.

We are going to collect the water quality data for the lakes of Washington State from the following the website:
https://green2.kingcounty.gov/lakes/Query.aspx. We are collecting the data from 01/01/1994 till 09/30/2022. For each lake there are multiple
monitoring stations, and each have a datasheet for the water quality parameters. For example, Lake Sammamish has the following monitoring stations:
0611, 0612, 0614, 0617, 0622, 0625 and M621. We will be merging these datasets as they hold the water quality data for the same lake. 

The WQI values will be classified into five levels: 

## excellent (WQI=91-100), good (WQI=76-90), fair (WQI=51-75), poor(WQI=26-50), and very poor (WQI=0-25)

## A.	ETL Process:

1.	The present dataset is downloaded from the earlier mentioned website has 35 columns and 10039 rows. We reviewed the data manually and realized that many of the columns are irrelevant for this analysis and hence we dropped them. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(31).png?raw=true)

The 21 columns which are dropped listed below:

Depth (m); Ammmonia Nitrogen Qualifier*; Cond Qualifier*, DO Qualifier*, Ecoli Qualifier*, Fecal Coliform Qualifier*,Nitrate Nitrite Qualifier*, 
OP Qualifier*, pH Qualifier*,Temperature Qualifier*, TN Qualifier*, TN Qualifier*,TP Qualifier*, Total Alkalinity Qualifier*, AN MDL (mg/L), 
Cond MDL (µmhos/cm), DO MDL (mg/L), Nitrate Nitrite MDL (mg/L), OP MDL (mg/L), TN MDL (mg/L), TP MDL (mg/L), Total Alkalinity MDL (mg/L).

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(32).png?raw=true)

2.	Form the leftover 14 columns, we dropped another two more columns because they do not have sufficient data in the rows. The additional dropped
columns are: Ecoli, Total Alkalinity (mg/L). 

In the present dataset the datatype in “CollectDate” is string type, so we needed to change it to the datetime format using:

WQ_0611_df_1["CollectDate"] = pd.to_datetime(WQ_0611_df_1["CollectDate"])

3.	In the dataset a few samples were duplicates in the “SampleNum” column, which should be unique. So, we removed the duplicates using this code:

df_1 = df_1.drop_duplicates(subset=['SampleNum'])

4.	Then we checked the number of null values in each column:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(33).png?raw=true)

 5.	We could see there were still lots of null values in each column. After inspecting the data, we realized that for each month there are numerous
 value parameters collected at different water depths at a single sampling location. Some of those vales are present and some are missing. So, it
 wouldn’t be incorrect if we replaced these null values in a month with the mean of the remaining values. 
 
With the following line of code we replaced the null values with the mean of the remaining values in that month. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(34).png?raw=true)

6.	We checked the null values again and summary of the null values in each column. We got the null values because in few months there was no 
readings, hence we could not get any mean value in that month which resulted in the null values.

7.	This time we dropped the rows with null values.

8.	The next step was to calculate the WQI. We referred the Brown et al., 1970 to calculate the WQI. The WQI needed to be calculated for each row.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(16).png?raw=true)

References: (Brown, R.M.; McClelland, N.I.; Deininger, R.A.; Tozer, R.G. A water quality index-do we dare. Water Sew. Work. 1970, 117, 339–343)

9.	One of the team members, Nicole Sanchez, applied this formula to the cleaned dataset using python in Jupyter Notebook. One more column 
was added (WQI) and the final shape of the dataset is:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(35).png?raw=true)

10.	The data frame was exported as “0611_lake_Sammamish.csv” in resources folder. Now the ETL was completed for one sampling point dataset.

11.	We applied similar ETL steps for the 6 other sampling point’s datasets (0611, 0612, 0614, 0617, 0622, 0625 and M621). Now, we had 7 final 
csv files.


## B.	Processing and exploratory analysis:

1.	In the previous section we were able to get 7 csv file which are the clean datasets and ready to be processed. So, we merged the seven csv file in to one with the following code.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(17).png?raw=true)

2.	We took a quick look at the Data Structure with the following code:

Df.head(), df.shape, df.info, df.dtypes(), df.value_counts(), df.describe
Another quick way to get feel of the data was to plot a histogram for each numerical attribute. To plot all the features in single view we used
the following line of code:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(18).png?raw=true)

The histogram shows the number of instances that have a given value in the range. The hist() method relies on Matplotlib, which in turn relies 
on a user-specified graphical backend to draw on the screen. So, before we plot anything, we need to specify which backend Matplotlib should use. 
The simplest option is to use Jupyter’s magic command %Matplotlib inline. This tells Jupyter to set up Matplotlib so it uses Jupyter’s own 
backend, plots, then renders within the notebook itself. 

The histogram plots give the visualization of the distribution and the median of the attributes. Many histogram are tail heavy, means they extend
much farther to the right of the median than to left. This may make it a bit harder for machine learning algorithms to detect patterns. We need 
to try transforming these attributes later on to have more bell-shaped distributions.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(19).png?raw=true)

3.	We visualized the counts in WQI using: WQI_df["WQI"].plot.density()

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(20).png?raw=true)

4.	So, in this dataset there is one dependent target variable i.e., “WQI” and 10 independent features. All of these 10 input variable feature 
do not have same amount of impact on the WQI, hence, to find out the extent of correlation of the input variable we are going to perform multiple
correlation analysis. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(21).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(22).png?raw=true)

We plotted the scatter plot for each feature with the target variable to check the relationship between the two. This visualization is a way 
to check if there is any relationship pattern between the two variables. The "WQI" is on the x-axis and the features are on the y-axis. 
The correlation graphs can tell us if the relationship of each feature with WQI is strong or weak.
With the data exploratory analysis and visualization we can identify the quirks in the data and may be clean it up before feeding the data to
machine learning algorithm.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(23).png?raw=true)

During the statistical exploratory analysis we faced the difficulty in plotting the scatter plots of all the features with the target variable in a grid format, but with the help of a class TA, we were able to sort that out.


## C.	Machine Learning Models

#### ML Model Construction:

To construct the ML model for prediction of water quality index, we need to select the proper input variables which have sufficient underlying information to predict WQI. Selection of appropriate input factors is very important as it could improve the model accuracy by eliminating the factors with undesirable impact on the predictive performance. 

For ANN, first plot the data to check if there is any outlier.

Throughout this implementation walkthrough, take some time to critically think about the following:

•	What about this dataset makes it complex? Is it a variable? Is it the distribution of values? Is it the size of the dataset?

•	Which variables should I investigate prior to implementing my model? What does the distribution look like? Hint: Use Pandas' Series.plot.density() method to find out.

•	What outcome am I looking for from the model? Which activation function should I use to get my desired outcome?

•	What is my accuracy cutoff? In other words, what percent testing accuracy must my model exceed?

After performing the exploratory data analysis, we have a better understanding of the kind of data we are dealing with. At this point we are 
ready to employ machine learning models to our data. 

•	The first step is to frame the problem and find an objective. 

Here we wanted to predict the water quality index (WQI) as a target variable which is dependent on independent feature variables such as using temperature, pH, turbidity, dissolved oxygen (DO), biochemical oxygen demand (BOD), and concentrations of other pollutants). Once, we know the 
WQI, we want to predict it for future years. So, we are going to employ two machine learning models:

###  Supervised regression ANN model.
###  Time series forecasting model.

•	This is a multiple regression univariate problem in which we are trying to predict a single value for each month. There is no continuous flow of
data coming into the system, and there is no particular need to do the adjustment to this data and is small enough to fit in memory, hence it is a
case of plain batch learning. 

•	For a regression problem, the following are the typical ways to measure the performance of the model: “Mean Squared Error”, “Residual Sum of
Squares” and “Mean Absolute Error”. These methods measure the distance between two vectors, the vectoe of prediction and vector of target values.
In this project we used “Mean Squared Error”and “Residual Sum of Squares” method to measure the performance of my machine learning models. 

Our Data Analysis Pipeline:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(61).png)

## D.	ANN regression model 

•	We applied the ANNregression model to predict the WQI and I used the following resources:

    	Scikit-Learn
    
    	Pandas
    
    	TensorFlow
    
    	Keras


#### •	 Data Preprocessing

We dropped the first column “CollectDate” from  the data frame as the machine learning models accepts only numerical values. Then, we assigned 
the target and features arrays to its respective variables and performed the splitting of the preprocessed data in to training and testing datasets.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(24).png?raw=true)

#### •	Compile, Train and Evaluate the Model

    	We defined the input features and hidden nodes for each hidden layers for deep neural network model. 
    	We defined checkpoint path and the filenames.
    	Complied the model with loss as binary_crossentropy and optimizer as “adam”.

 In the development of ANN model we faced some difficulty, while assigning the values to the x, and y variable, the model code run was giving 
 error that the "Expected 2D array, got 1D array instead. I resolved it with mentioning the ".reshape(-1,1). After a few trials the ANN model 
 run was successful and giving low MSE values as shown in the screenshot. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(25).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(26).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(27).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(28).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(29).png?raw=true)


## E.	Time Series Forecasting Model

A time series is a succession of chronologically ordered data spaced at some specified intervals. The forecasting means predicting the future 
value of a time series by using models to study its past behavior (autoregressive).

#### Skforecast ML Model:

•	We chose the “SKforecast model” for the time series forecasting with Python and Scikit-learn libraries based on the TA’s recommendation for 
its simplicity. Skforecast is a simple library that contains the classes and functions necessary to adapt any Scikit-learn regression model to
forecasting problems.

•	To apply machine learning models to forecasting problems, the time series need to be transformed into a matrix in which each value is related 
to the time window that precedes it. Hence, we created a dataframe with time index and “WQI” column and removed all the repeated values of
time/date. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(63).png)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/mean.png)

•	We had difficulty with this model as the date index in the dataset was not measured in fixed regular intervals, but on different dates. So, 
to sort out this problem, we had to make an assumption that the date for observations are all on the first day of the month. With the help of
the class Instructor “Khaled Karman“ we were able to construct a new column with the date with first day of the month. So, we replaced the 
actual time/date index with the new one which we created for the first day of the month and dropped the old index from the dataset. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/date_change.png)

•	Now, we have a time series for “WQI” with monthly date of observation between 1994 and 2008, indented to create an auto regressive model 
capable of predicting future monthly “WQI”.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(65)_Copy.png)

•	The date in the index should be stored in datetime format and since the data is monthly, the frequency is set as the Monthly Started ‘MS’.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/frequency.png)

•	Then we split the data into train and test sets. Mention the number of steps as the duration of months as the test set to evaluate the predictive 
capacity of the model.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Split_and%20train.png)

•	With the ForecasterAutoreg class, we created a  model and trained it with a “LinearRegressor” regressor with a time window of 12 lags. 
This means that the model uses the previous 12 months as predictors.

•	So, the model ran successfully, but the predicted results were not in the right format. The index of the output table was overwritten with a
RangeIndex of step 1 and not in the continuation of the training dataset. So, we had to write the following code mentioned in the screenshot 
to make it in correct datetime index.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Skpredictions.png)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Model.png)

•	Once the model is trained, the test data is predicted for 60 months into the future.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(72)_copy.png)

•	We evaluated the performance of this model with the “Mean Squared Error” value which produced a value of 820, which is a huge number for mse. 
It means that the performance of the model is poor and the model needed some improvement. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/mse.png)

•	To improve the model Hyperparameter tuning could be used, which means changing a few parameters of the model and re-evaluating the performance. 
We plan to improve this model in coming months.

## ARIMA ML Model

•	When we worked with the Skforecast model, we were unable to move further as we faced some difficulties with the code. So, we wanted to explore
other methods for time series forecasting.  Hence, we researched a bit on the internet and found ARIMA model for time series forecasting.
But there was one condition to this model that it can not be applied to the data which is not stationary. Hence, we needed to check for the
stationarity of the dataset.

•	There are different components of time series data, most of the time series has Trend, Seasonality, Irregularity, cyclic variation associated 
with them. Any time series may be split into the following components: Base Level + Trend + Seasonality + Error. 

•	The following screenshot shows the trend of the dataset.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/trend.png)

•	A trend is observed when there is an increasing or decreasing slope observed in the time series. Whereas seasonality is observed when there is 
a distinct repeated pattern observed between regular intervals due to seasonal factors. It could be because of the month of the year, the day of 
the month, weekdays or even time of the day.
•	However, It is not mandatory that all time series must have a trend and/or seasonality. A time series may not have a distinct trend but have 
a seasonality. The opposite can also be true. So, a time series may be imagined as a combination of the trend, seasonality and the error terms.
•	Before performing the time series analysis first we need to check the stationarity of the time series data.Time series data analysis require 
that the time series data to be stationary.

•	How to check if the data is stationary or not:

1.	Constant mean
2.	Constant Variance
3.	Autocovariance that does not depend on time.

If all these conditions are met than we can apply the time series analysis.

•	To check the stationarity there are two popular test:

1.	Rolling Statistics: It is a visual technique and we plot the moving average and variance to see if its varying with time.

3.	Augmented Dicky-fuller Test (ADCF): this is a statistical test to check the stationarity.

#### Null hypothesis: The time series is non-stationary in nature.
#### Alternate hypothesis: The time series is stationary in nature.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/stationarity_test.png)

The test results comprise of a Test Statistic and some Critical values. 
If the (Test Statistics) < (Critical values) and when p value is less than 0.05.
We reject the null hypothesis and say that the series is stationary.
In our time series data here are the statistics:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/dicky_test_results.png)

So, we can see that the test statistic is greater than all the critical values at different significance levels. It means we failed to reject the null hypothesis and we can say that this time series data is non-stationary.

•	Hence, we needed to make the data stationary by performing transformations to it. So, we changed the time series data to log scale and performed
the stationarity test again and it failed the test again.

•	Next, we are going to get the difference between the moving average and the actual WQI value. 
We are performing this exercise to make the transformations to the time series data so that it can become stationary to make it eligible for 
the time series analysis and predictions. So, we subtracted the moving average from the log scale dataset and checked the stationarity. 

•	We defined a function test_stationary(timeseries), to check the stationarity of the transformed data with visualizations and statistic tests.
Using this defined function we checked the stationarity of the transformed data with difference between the moving average and the actual WQI 
value. And got the following results:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Passed_Dicky.png)

This time  (Test Statistics) < (Critical values) and when p value is less than 0.05.
Hence, we reject the null hypothesis and say that the transformed time series data “datasetLogScaleMinusMovingAverage” is stationary.

•	Now, we are ready to apply the ARIMA Model. AR+MA: "AR" is the Auto regressive part of the model, "MA" is the Moving average part of the model, 
and I is the integration of the two. AR gives autoregressive lags i.e. P, MA gives the moving average i.e. Q and I is the integration is d 
(order of differentiation). To predict the “P” we need to plot PACF graph, to predict Q plot ACF graph. 

ARIMA model in words:

Predicted Yt = Constant + Linear combination Lags of Y (upto p lags) + Linear Combination of Lagged forecast errors (upto q lags)

The objective, therefore, is to identify the values of p, d and q. With the code and graph mentioned in the screenshot, we were able to find the 
p and q values to substitute it in the ARIMA model.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/p_q%26d.png)

•	Before applying ARIMA model we split the data into train and train dataset.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Split_ARIMA.png)

•	Imported a few libraries and applied the ARIMA model as follows:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/ARIMA_Model.png)

•	The following graph shows the test dataset and predicted dataset with its performance value as residual squared sum.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Test_predicted_ARIMA.png)

•	The following graph shows the train dataset and predicted dataset with its performance value as residual squared sum.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Test_predicted_ARIMA.png)

•	The model run was successful with better performance than the Skforecast model. However, we found this model to be more complex to understand.
After getting the predicted results, it requires us to do the inverse the transformation which we originally applied to the dataset to make it
stationary. However, we would like to keep exploring more about it to get deeper understanding of ARIMA model and apply it in different time series
forecasting.

## Dashboard

Goal for Segment 2: 
    -	Visualization between lake parameters over time from 1994 to 2008
    -	Frequency of parameter over time 
    -	Depth visualization (optional) since depth was removed during data cleaning process. 

This is the visual that shows each parameter over a period of time. As we can see, the peaks are during 1996 and 2006. 

![Parameter over time](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Code%20and%20Visual%20Snippets/Parameters_Over_Time.png)

This is the visual heatmap between the average number of each parameter over a period of time. The darker the color, the higher the number. The 
main point is to compare the mean of the parameters with the mean of water quality index from 1994 to 2008.

<img width="1196" alt="Heatmap" src="https://user-images.githubusercontent.com/106849689/204072376-90bcf68f-7c3a-4c77-b40d-da7c4370ac6b.png">

Frequency of distinct time each number of parameters appear during those months from 1994 to 2008.

![Frequency](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Code%20and%20Visual%20Snippets/Frequency%20of%20AmmoniaNitrogen.png)

Problem from Segment 2:

-	The heatmap doesn’t relate nor add to the story of the main project. 
-	Need to show a correlation as scatter plot. 

Goal for Segment 3: 

-	Create more scatter plots with linear regression line with all the parameters.

![Correlation between Ammonia and WQI](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Segment_3/Corr%20Ammonia%20vs%20WQI.png)

This graph shows the correlation between Ammonia Nitrogen and WQI. The R-Squared is 0.0038556 and P-value is 0.390971. As we can see, the correlation in this graph is low since the plot is very scattered away from the linear line. 

 | Parameters Correlation | Parameters Correlation | Parameters Correlation |
| :---:         |     :---:      |    :---:   |
| ![Corr1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Segment_3/Correlation(1).png)   | ![Corr2](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Segment_3/Correlation(2).png)     | ![Corr3](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Segment_3/Correlation(3).png)    |

In these correlations between all parameters with WQI, we can see Dissolved Oxygen, Fecal Coliform, Nitrate Nitrite and Total Nitrogen have a
stronger correlation with WQI accodring to the graph since the plot density is more compared to other parameters such as Ammonia, Conductivity,
Orthphosphate Phosphate, PH, Temperature and Total Phosphorus. 

This is where you can find the [Tableau link](https://public.tableau.com/app/profile/ling7171/viz/CorrelationatLakeSammamish/WQI) for further details. 


## Website

- First, we created a simple website that:
    - displays our logo.
    - contains a drop-down list of the lakes we are using to calculate the water quality index.
    - displays a map of the lake selected from the drop-down list.
    
- We updated the website by:
    - adding more formatting to the text and background.
    - replacing the map images with interactive maps.
    
<img width="898" alt="WQI_Pred_Webpage" src="https://user-images.githubusercontent.com/106849689/204072178-30292904-a6d0-425f-940a-59b7fe9f91ee.png">

- We finalized the website to include the WQI forecasts for the last 2 lakes.

![final_website_pic](https://user-images.githubusercontent.com/106849689/204072781-c7bf68f9-e776-4abc-bd39-901d53b077d0.png)

- Final website link:
https://nsanchez76.github.io/

## Presentation

 Link:
 https://docs.google.com/presentation/d/16ZrQ_KEqKrhyH9IsrkqKFNsgMb9nEmqp7MNBwoZUki4/edit?usp=sharing

### Segment 1

- We created the initial draft of our presentation:
    - the reason why they selected this topic.
    - the description of the source of data.
    - the questions we hope to answer with the data.

### Segment 2

- We updated the presentation to add:
    - a description of how we, as a team, explored the data.
    - a desctiption of how we, as a team, analyzed the data.

### Segment 3

- We worked as a team to update the presentation with their own visuals for the slides they are
  presenting.
- We, as a team, created slides that showcased the technologies, languages, tools, and algorithms
  used throughout the project

### Segment 4

- We updated the presentation to:
    - include the result of analysis.
    - add recommendation for future analysis.
    - describe anything the team would have done differently.


## Conclusion and Recommendations

Group 9 completed all requirements according to the rubric for the final project. We downloaded raw data. Explored the data using MS-Excel. Cleaned the data. 
Created a dashboard in Tableau to further explore the data. Created a database to save the cleaned data. Created a machine learning model. Ran the clean data through
the model to test the water quality prediction accuracy. Created a forecasting model. Ran the clean data through the model to forecast the water quality index for
the next 12 months. Finally, created a website to allow users to select from 3 different lakes and view the water quality index for the next 12 months.

Recommendations for improvements:
- Add date selector element to webpage
- Make webpage tables and graphs interactive
- Improve the forecasting model for lower MSE values.
- Apply same methods to other bodies of water

