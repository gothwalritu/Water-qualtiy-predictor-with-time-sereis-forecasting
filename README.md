# Final_Project_UCB_Bootcamp


# Using Machine Learning Models for Predicting Water quality Index of Lakes in Washington state

Water Quality Index: The water quality index (WQI) is extensively used to assess and classify the quality of surface water and groundwater. The water quality index is computed based on the physicochemical parameters of the water such as, temperature, pH, turbidity, dissolved oxygen (DO), biochemical oxygen demand (BOD), and concentrations of other pollutants), for the estimation of water quality.  WQI provides a meaningful way to categorize the quality of any water resource in some quantitative form, which could help decision makers and planners to make well informed decision in the subject of water resources management. However, it involves lengthy calculations to formulate the water quality index and hence, requires a lot of resources in terms of time and efforts. To solve this problem an alternative approach is needed, which is more efficient and accurate to estimate the WQI.


Machine learning (ML) have proven to be a cutting edge tool for modeling complex non-linear behaviors in water resources research and has been also used for assessing the water quality. However, there are many techniques in ML which could be applied to estimate and predict the WQI. In the first section of the project we would like to explore the different techniques of machine learning and compare them with each other in terms of its accuracy. We would like to choose the one technique which is most suitable for the kind of data we have in order to predict the WQI of lakes in Washington state. We are going to refer the article Khoi et al., (2022) for choosing the approach to perform the ML.


We are going to collect the water quality data for the lakes of Washington State from the following the website: https://green2.kingcounty.gov/lakes/Query.aspx. We are collecting the data from 01/01/1994 till 09/30/2022. For each lake there are multiple monitoring stations, and each have a datasheet for the water quality parameters. For example, Lake Sammamish has the following monitoring stations: 0611, 0612, 0614, 0617, 0622, 0625 and M621. We will be merging these datasets as they hold the water quality data for the same lake. 

## The WQI values will be classified into five levels: 

excellent (WQI=91-100), good (WQI=76-90), fair (WQI=51-75), poor(WQI=26-50), and very poor (WQI=0-25)

## ML Model Construction:

To construct the ML model for prediction of water quality index, first we need to select the proper input variables which has sufficient underlying information to predict WQI. Selection of appropriate input factors is very important as it could improve the model accuracy by eliminating the factors with undesirable impact on the predictive performance. 

For ANN, first plot the data to check if there is any outlier.

Throughout this implementation walkthrough, take some time to critically think about the following:

•	What about this dataset makes it complex? Is it a variable? Is it the distribution of values? Is it the size of the dataset?

•	Which variables should I investigate prior to implementing my model? What does the distribution look like? Hint: Use Pandas' Series.plot.density() method to find 
out.

•	What outcome am I looking for from the model? Which activation function should I use to get my desired outcome?

•	What is my accuracy cutoff? In other words, what percent testing accuracy must my model exceed?


## ETL Process:

In the present dataset which we have collected from the earlier mentioned website we have 35 columns. We reviewed the data manually and realized that many of the columns are irrelevant for this analysis and hence we dropped them. 

The 21 columns which we have dropped are listed below:

Depth (m); Ammmonia Nitrogen Qualifier*; Cond Qualifier*, DO Qualifier*, Ecoli Qualifier*, Fecal Coliform Qualifier*,Nitrate Nitrite Qualifier*, OP Qualifier*, pH Qualifier*,Temperature Qualifier*, TN Qualifier*, TN Qualifier*,TP Qualifier*, Total Alkalinity Qualifier*, AN MDL (mg/L), Cond MDL (µmhos/cm), DO MDL (mg/L), Nitrate Nitrite MDL (mg/L), OP MDL (mg/L), TN MDL (mg/L), TP MDL (mg/L), Total Alkalinity MDL (mg/L).

Form the leftover 14 columns, we dropped another two more columns because they are not having sufficient data in the rows. Dropped columns are: Ecoli, Total Alkalinity (mg/L). 

In the present dataset the datatype in “CollectDate” is string type, we changed it to datetime format using:
WQ_0611_df_1["CollectDate"] = pd.to_datetime(WQ_0611_df_1["CollectDate"])

In the dataset few samples were repeated as there were duplicates in the column “SampleNum” which should be unique. So, we removed the duplicates using this code:
df_1 = df_1.drop_duplicates(subset=['SampleNum'])



Then we checked the number of null values in each column:

df_1.isnull().sum()

(10039, 12)

Out[53]:
CollectDate                           0
SampleNum                             0
Ammmonia Nitrogen (mg/L)           7495
Conductivity (µmhos/cm)            1904
Dissolved Oxygen (mg/L)            1987
Fecal Coliform (CFU/100ml)         9723
Nitrate Nitrite (mg/L)             7469
Orthophosphate Phosphate (mg/L)    7560
pH                                 1894
Temperature (°C)                   1884
Total Nitrogen (mg/L)              7392
Total Phosphorus (mg/L)            7289
dtype: int64


We can see there are still lots of null values in each column. After observing the data we realized that for each month there are numerous value parameters collected at different water depth at a single location. And some of those vales are present and some are missing. So, it won’t be incorrect if we replace these null values in a month with the mean of the remaining values. 

With the following line of code we replaced the null values with the mean of the remaining values in that month. 

df_3 = df_1.groupby(["CollectDate"]).apply(lambda x:x.fillna(x.mean()))

We checked the null values again and summary of the null values in each column is as follows:

CollectDate                           0
SampleNum                             0
Ammmonia Nitrogen (mg/L)           1477
Conductivity (µmhos/cm)              68
Dissolved Oxygen (mg/L)              80
Fecal Coliform (CFU/100ml)         6991
Nitrate Nitrite (mg/L)             1358
Orthophosphate Phosphate (mg/L)    1455
pH                                 1881
Temperature (°C)                     49
Total Nitrogen (mg/L)              1413
Total Phosphorus (mg/L)              79


We got these null values because in few months there was no reading and hence we could not get any mean value in that month which resulted in the null values.

This time we dropped the rows with null values and the final shape of the data frame is:
(2546, 12)
Out[49]:
CollectDate                        2546
SampleNum                          2546
Ammmonia Nitrogen (mg/L)           2546
Conductivity (µmhos/cm)            2546
Dissolved Oxygen (mg/L)            2546
Fecal Coliform (CFU/100ml)         2546
Nitrate Nitrite (mg/L)             2546
Orthophosphate Phosphate (mg/L)    2546
pH                                 2546
Temperature (°C)                   2546
Total Nitrogen (mg/L)              2546
Total Phosphorus (mg/L)            2546
dtype: int64

The final data frame is exported as “0611_lake_Sammamish.csv” in resources folder.

The next step is to calculate the WQI. 

With the remaining 12 columns we kept 12 input variables. All of these 12 input variable do not have same amount of impact on the WQI and hence, to find out the extent of correlation of the input variable we are going to perform multiple correlation analysis. The input variable with the highest value of R2 will be listed and the input variable with small R2 value could be dropped before performing the ML exercise. Based on the R2 values we can choose the combination of input variables to perform the ML to predict WQI and compare the accuracy of each method.
Download the data.

Take a quick look at the Data Structure with the following code:

Df.head(), df.shape, df.info, df.dtypes(), df.value_counts(), df.describe

Another quick way to get feel of the data is to plot a histogram for each numerical attribute. To plot all the features in single go we are using the following line of code:

%matplotlib inline

import matplotlib.pyplot as plt

WQI_df.hist(bins=50, figsize=(20,15))

plt.show()

Histogram shows the number of instances that have a given value range. The hist() method relies on Matplotlib, which in turn relies on a user-specified graphical backend to draw on the screen. So, before we plot anything, we need to specify which backend Matplotlib should use using jupyter's magic command to specify which backend Matplotlib should use. The simplest option is to use Jupyter’s magic command %Matplotlib inline. This tells Jupyter to set up Matplotlib so it uses Jupyter’s own backend. Plots then rendered within the notebook itself. 

