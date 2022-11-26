# Water Quality Index Final Project
## Group 9: Ritu Gothwal, Erica Gutierrez, Ling Hoang Do, Nicole Sanchez

## Introduction

We all love recreation activities in and around water. We always wonder where we should go, and which locations have the safest water for
recreational activities, as well as which locations have potable water.

In this project we will develop a tool which will show locations around Lake Sammamish in Washington State, and display which are safe for
recreational and potability use. It will be an interactive tool which takes user input for geolocation and recommends the locations which 
are fit for potability and recreational uses. The location will be displayed using marker layer map and pop-up markers, information about 
weather and water quality.

## ERD

We used Quick DBD to create a ERD of our data
![QuickDBD-export](https://user-images.githubusercontent.com/106849689/200139417-deccad9a-fb82-4ff2-92a1-862dc4d4b325.png)


## Using Machine Learning Models for Predicting Water quality Index of Lakes in Washington state

## Water Quality Index:

The water quality index (WQI) is extensively used to assess and classify the quality of surface water and groundwater. The water quality index is computed based on the physicochemical parameters of the water such as, temperature, pH, turbidity, dissolved oxygen (DO), biochemical oxygen demand (BOD), and concentrations of other pollutants), for the estimation of water quality.  WQI provides a meaningful way to categorize the quality of any water resource in some quantitative form, which could help decision makers and planners to make well informed decision in the subject of water resources management. However, it involves lengthy calculations to formulate the water quality index and hence, requires a lot of resources in terms of time and efforts. To solve this problem an alternative approach is needed, which is more efficient and accurate to estimate the WQI.

Machine learning (ML) have proven to be a cutting edge tool for modeling complex non-linear behaviors in water resources research and has been also used for assessing the water quality. However, there are many techniques in ML which could be applied to estimate and predict the WQI. In the first section of the project we would like to explore the different techniques of machine learning and compare them with each other in terms of its accuracy. We would like to choose the one technique which is most suitable for the kind of data we have in order to predict the WQI of lakes in Washington state. We are going to refer the article Khoi et al., (2022) for choosing the approach to perform the ML.




We are going to collect the water quality data for the lakes of Washington State from the following the website: https://green2.kingcounty.gov/lakes/Query.aspx. We are collecting the data from 01/01/1994 till 09/30/2022. For each lake there are multiple monitoring stations, and each have a datasheet for the water quality parameters. For example, Lake Sammamish has the following monitoring stations: 0611, 0612, 0614, 0617, 0622, 0625 and M621. We will be merging these datasets as they hold the water quality data for the same lake. 

The WQI values will be classified into five levels: 

## excellent (WQI=91-100), good (WQI=76-90), fair (WQI=51-75), poor(WQI=26-50), and very poor (WQI=0-25)

## A.	ETL Process:

1.	The present dataset is downloaded from the earlier mentioned website has 35 columns and 10039 rows. I reviewed the data manually and realized that many of the columns are irrelevant for this analysis and hence I dropped them. 


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(31).png?raw=true)

The 21 columns which are dropped listed below:

Depth (m); Ammmonia Nitrogen Qualifier*; Cond Qualifier*, DO Qualifier*, Ecoli Qualifier*, Fecal Coliform Qualifier*,Nitrate Nitrite Qualifier*, OP Qualifier*, pH Qualifier*,Temperature Qualifier*, TN Qualifier*, TN Qualifier*,TP Qualifier*, Total Alkalinity Qualifier*, AN MDL (mg/L), Cond MDL (µmhos/cm), DO MDL (mg/L), Nitrate Nitrite MDL (mg/L), OP MDL (mg/L), TN MDL (mg/L), TP MDL (mg/L), Total Alkalinity MDL (mg/L).

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(32).png?raw=true)

2.	Form the leftover 14 columns, I dropped another two more columns because they are not having sufficient data in the rows. Dropped columns are: Ecoli, Total Alkalinity (mg/L). 

In the present dataset the datatype in “CollectDate” is string type, we changed it to datetime format using:

WQ_0611_df_1["CollectDate"] = pd.to_datetime(WQ_0611_df_1["CollectDate"])

3.	In the dataset few samples were in repetition as there were duplicates in the column “SampleNum”, however, which should be unique. So, I removed the duplicates using this code:

df_1 = df_1.drop_duplicates(subset=['SampleNum'])


4.	Then I checked the number of null values in each column:


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(33).png?raw=true)


 5.	I could see there were still lots of null values in each column. After observing the data, I realized that for each month there are numerous value parameters collected at different water depth at a single location. And some of those vales are present and some are missing. So, it won’t be incorrect if we replace these null values in a month with the mean of the remaining values. 
 
With the following line of code we replaced the null values with the mean of the remaining values in that month. 


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(34).png?raw=true)

6.	I checked the null values again and summary of the null values in each column. I got the null values because in few months there was no reading and hence we could not get any mean value in that month which resulted in the null values.


7.	This time we dropped the rows with null values.

8.	The next step is to calculate the WQI. I referred the Brown et al., 1970 to calculate the WQI. The WQI needed to be calculated for each row.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(16).png?raw=true)

References: (Brown, R.M.; McClelland, N.I.; Deininger, R.A.; Tozer, R.G. A water quality index-do we dare. Water Sew. Work. 1970, 117, 339–343)

9.	So, one of the team member “Nicole Sanchez” applied this formula to the cleaned dataset using python in Jupyter Notebook. So one more column is added now and the final shape of the dataset is:


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(35).png?raw=true)

10.	The data frame is exported as “0611_lake_Sammamish.csv” in resources folder. Now the ETL is completed for one sampling point dataset.

11.	I did the similar ETL steps for the 6 other sampling point’s dataset (0611, 0612, 0614, 0617, 0622, 0625 and M621). And now we have 7 final csv files.


## B.	Processing and exploratory analysis:

1.	In the previous section I was able to get 7 csv file which are the clean datasets and ready to be processed. So, I merged the seven csv file in to one with the following code.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(17).png?raw=true)

2.	A quick look at the Data Structure with the following code:

Df.head(), df.shape, df.info, df.dtypes(), df.value_counts(), df.describe
Another quick way to get feel of the data, I plotted a histogram for each numerical attribute. To plot all the features in single go we are using the following line of code:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(18).png?raw=true)

Histogram shows the number of instances that has a given value range. The hist() method relies on Matplotlib, which in turn relies on a user-specified graphical backend to draw on the screen. So, before we plot anything, we need to specify which backend Matplotlib should use using jupyter's magic command to specify which backend Matplotlib should use. The simplest option is to use Jupyter’s magic command %Matplotlib inline. This tells Jupyter to set up Matplotlib so it uses Jupyter’s own backend. Plots then rendered within the notebook itself. 

The histogram plots gives the visualization of the distribution and the median of the attributes. Many histogram are tail heavy, means they extend much farther to the right of the median than to left. This may make it a bit harder for machine learning algorithms to detect patterns. I need to try transforming these attributes later on to have more bell-shaped distributions.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(19).png?raw=true)

3.	I visualized the counts in WQI using: WQI_df["WQI"].plot.density()

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(20).png?raw=true)

4.	So, in this dataset there is one dependent target variable i.e., “WQI” and 10 independent features. All of these 10 input variable feature do not have same amount of impact on the WQI and hence, to find out the extent of correlation of the input variable we are going to perform multiple correlation analysis. 


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(21).png?raw=true)


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(22).png?raw=true)


I plotted the scatter plot for each feature with the target variable to check the relationship between the two. It is the visualization way to check if there is any relationship pattern between the two variables. The "WQI" is on the x-axis and the features are on the y-axis. 
The correlation graphs tell us if the relationship of each feature with WQI is strong or weak.
With the data exploratory analysis and visualization we can identify the quirks in the data and may be clean it up before feeding the data to machine learning algorithm.

 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(23).png?raw=true)


During the statistical exploratory analysis I faced the difficulty in plotting the scatter plots of all the features with the target variable in a grid format, but with the help of TA I was able to sort that out.


## C.	Machine Learning Models

#### ML Model Construction:

To construct the ML model for prediction of water quality index, we need to select the proper input variables which has sufficient underlying information to predict WQI. Selection of appropriate input factors is very important as it could improve the model accuracy by eliminating the factors with undesirable impact on the predictive performance. 

For ANN, first plot the data to check if there is any outlier.

Throughout this implementation walkthrough, take some time to critically think about the following:

•	What about this dataset makes it complex? Is it a variable? Is it the distribution of values? Is it the size of the dataset?

•	Which variables should I investigate prior to implementing my model? What does the distribution look like? Hint: Use Pandas' Series.plot.density() method to find out.

•	What outcome am I looking for from the model? Which activation function should I use to get my desired outcome?

•	What is my accuracy cutoff? In other words, what percent testing accuracy must my model exceed?

After performing the exploratory data analysis, I have a better understanding of the kind of data I am dealing with. At this point I am ready to employ machine learning models to our data. 

•	The first step is to frame the problem and find an objective. 

So, here I wanted to predict the water quality index (WQI) as a target variable which is dependent on independent feature variables such as using temperature, pH, turbidity, dissolved oxygen (DO), biochemical oxygen demand (BOD), and concentrations of other pollutants). Once, I know the WQI I want to predict it for future years. So, I am going to employ two machine learning models:

###  Supervised regression ANN model.
###  Time series forecasting model.

•	This is a multiple regression univariate problem as I am trying to predict a single value for each month. There is no continuous flow of data coming into the system, and there is no particular need to do the adjustment to this data and is small enough to fit in memory, hence it is a case of plain batch learning. 

•	For a regression problem following are the typical ways to measure the performance of the model: “Mean Squared Error”, “Residual Sum of Squares” and “Mean Absolute Error”. These methods measure the distance between two vectors, the vectoe of prediction and vector of target values. In this project I am using “Mean Squared Error”and “Residual Sum of Squares” method to measure the performance of my machine learning models. 

Develop a Data Analysis Pipelines: ??

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(61).png)



## D.	ANN regression model 

•	I applied the ANNregression model to predict the WQI and I used the following resources:

    	Scikit-Learn
    
    	Pandas
    
    	TensorFlow
    
    	Keras


#### •	 Data Preprocessing

I dropped the first column “CollectDate” from  the data frame as the machine learning models accepts only numerical values. Then, I assigned the target and features arrays to its respective variables and performed the splitting of the preprocessed data in to training and testing datasets.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(24).png?raw=true)

#### •	Compile, Train and Evaluate the Model

    	I defined the input features and hidden nodes for each hidden layers for deep neural network model. 
    	I defined checkpoint path and the filenames.
    	Complied the model with loss as binary_crossentropy and optimizer as “adam”.

 In the development of ANN model I faced some difficulty, while assigning the values to the x, and y variable, the model code run was giving error that the "Expected 2D array, got 1D array instead. I resolved it with mentioning the ".reshape(-1,1). After a few trials the ANN model run was successful and giving low MSE values as shown in the screenshot. 


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(25).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(26).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(27).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(28).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(29).png?raw=true)


## E.	Time Series Forecasting Model

A time series is a succession of chronologically ordered data spaced at some specified intervals. The forecasting means predicting the future value of a time series by using models to study its past behavior (autoregressive).

#### Skforecast ML Model:

•	I choose “SKforecast mode” for time series forecasting with Python and Scikit-learn libraries based on the TA’s recommendation for its simplicity. Skforecast is a simple library that contains the classes and functions necessary to adapt any Scikit-learn regression model to forecasting problems.

•	To apply machine learning models to forecasting problems, the time series need to be transformed into a matrix in which each value is related to the time window that precedes it. Hence, I created a dataframe with time index and “WQI” column and removed all the repeated values of time/date. 


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(63).png)


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/mean.png)

•	I had difficulty with this model as the date index in the dataset was not on the fixed regular intervals, but on different dates. So, to sort this problem I had to make an assumption that the date for observations are all on the first day of the month. With the help of Instructor “Khaled “ I was able to construct a new column with the date with first day of the month. So, I replaced the actual time/date index with the new one which I created for the first day of the month and dropped the old index from the dataset. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/date_change.png)

•	Now, I have a time series for “WQI” with monthly date of observation between 1994 and 2008, indented to create an auto regressive model capable of predicting future monthly “WQI”.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(65)_Copy.png)


•	The date in the index should be stored in datetime format and since the data is monthly, the frequency is set as the Monthly Started ‘MS’.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/frequency.png)

•	Then split the data into train and test. Mention the number of steps as the duration of months as the test set to evaluate the predictive capacity of the model.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Split_and%20train.png)


•	With the ForecasterAutoreg class, I created a  model and trained it with a “LinearRegressor” regressor with a time window of 12 lags. This means that the model uses the previous 12 months as predictors.

•	So, the model run successfully but the predicted results were not in the right format. The index of the output table was overwritten with a RangeIndex of step 1 and not in the continuation of the training dataset. So, I had to write the following code mentioned in the screenshot to make it in correct datetime index.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Skpredictions.png)



![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Model.png)


•	Once the model is trained, the test data is predicted for 60 months into the future.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(72)_copy.png)

•	I evaluated the performance of this model with the “Mean Squared Error” value. And it is producing the 820, which is a huge number for mse. It means that the performance of the model is poor and the model needed some improvement. 

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/mse.png)

•	To improve the model Hyperparameter tuning could be done, which means changing a few parameters of the model and re-evaluating the performance. I am planning to improve this model in coming months.



## ARIMA ML Model

•	When I was working with Skforecast model I was unable to move further as I was facing some difficulties with the code. So, I wanted to explore other methods for time series forecasting.  Hence, I researched a bit on the internet and found ARIMA model for time series forecasting.
But there was one condition to this model that it can not be applied to the data which is not stationary. Hence, I needed to check for the stationarity of the dataset.

•	There are different components of time series data, most of the time series has Trend, Seasonality, Irregularity, cyclic variation associated with them. Any time series may be split into the following components: Base Level + Trend + Seasonality + Error. 

•	The following screenshot is showing trend of the dataset.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/trend.png)

•	A trend is observed when there is an increasing or decreasing slope observed in the time series. Whereas seasonality is observed when there is a distinct repeated pattern observed between regular intervals due to seasonal factors. It could be because of the month of the year, the day of the month, weekdays or even time of the day.
•	However, It is not mandatory that all time series must have a trend and/or seasonality. A time series may not have a distinct trend but have a seasonality. The opposite can also be true. So, a time series may be imagined as a combination of the trend, seasonality and the error terms.
•	Before performing the time series analysis first we need to check the stationarity of the time series data.Time series data analysis require that the time series data to be stationary.


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
we reject the null hypothesis and say that the series is stationary.
In our time series data here are the statistics:


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/dicky_test_results.png)

So, we can see that the test statistic is greater than all the critical values at different significance levels. It means we failed to reject the null hypothesis and we can say that this time series data is non-stationary.

•	Hence, I needed to make the data stationary by performing transformations to it. So, I changed the time series data to log scale and performed the stationarity test again and it failed the test again.

•	Next, I am going to get the difference between the moving average and the actual WQI value. 
I am doing this exercise to make the transformations to my time series data so that it can become stationary to make it eligible for the time series analysis and predictions. So, I subtracted the moving average from the log scale dataset and checked the stationarity. 

•	I defined a function test_stationary(timeseries), to check the stationarity of the transformed data with visualizations and statistic tests. Using this defined function I checked the stationarity of the transformed data with difference between the moving average and the actual WQI value. And got the following results:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Passed_Dicky.png)


This time  (Test Statistics) < (Critical values) and when p value is less than 0.05.
Hence, we reject the null hypothesis and say that the transformed time series data “datasetLogScaleMinusMovingAverage” is stationary.

•	Now, I am ready to apply ARIMA Model. AR+MA: AR is the Auto regressive part of the model and MA, is the Moving average part of the model , I is the integration of the two. AR gives autoregressive lags i.e. P, MA gives the moving average i.e. Q and I is the integration is d (order of differentiation). To predict the “P” we need to plot PACF graph, to predict Q plot ACF graph. 

ARIMA model in words:

Predicted Yt = Constant + Linear combination Lags of Y (upto p lags) + Linear Combination of Lagged forecast errors (upto q lags)

The objective, therefore, is to identify the values of p, d and q. With the code and graph mentioned in the screenshot I was able to find the p and q values to substitute it in the ARIMA model.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/p_q%26d.png)

•	Before applying ARIMA model I splitted the data into train and train dataset.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Split_ARIMA.png)

•	Imported a few libraries and applied the ARIMA model as follows:

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/ARIMA_Model.png)

•	The following graph shows the test dataset and predicted dataset with its performance value as residual squared sum.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Test_predicted_ARIMA.png)


•	The following graph shows the train dataset and predicted dataset with its performance value as residual squared sum.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Test_predicted_ARIMA.png)

•	The model run was successful with better performance than the Skforecast model. However, I find this model to be more complex to my understanding. And the after getting the predicted results it requires us to do the inverse the transformation which we did to the dataset to make it stationary. However, I would like to keep exploring more about it to get deeper understanding of ARIMA model and apply it in different time series forecasting.

