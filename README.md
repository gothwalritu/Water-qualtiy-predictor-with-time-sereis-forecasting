# Final_Project_UCB_Bootcamp

## Using Machine Learning Models for Predicting Water quality Index of Lakes in Washington state

## Water Quality Index:

The water quality index (WQI) is extensively used to assess and classify the quality of surface water and groundwater. The water quality index is computed based on the physicochemical parameters of the water such as, temperature, pH, turbidity, dissolved oxygen (DO), biochemical oxygen demand (BOD), and concentrations of other pollutants), for the estimation of water quality.  WQI provides a meaningful way to categorize the quality of any water resource in some quantitative form, which could help decision makers and planners to make well informed decision in the subject of water resources management. However, it involves lengthy calculations to formulate the water quality index and hence, requires a lot of resources in terms of time and efforts. To solve this problem an alternative approach is needed, which is more efficient and accurate to estimate the WQI.

Machine learning (ML) have proven to be a cutting edge tool for modeling complex non-linear behaviors in water resources research and has been also used for assessing the water quality. However, there are many techniques in ML which could be applied to estimate and predict the WQI. In the first section of the project we would like to explore the different techniques of machine learning and compare them with each other in terms of its accuracy. We would like to choose the one technique which is most suitable for the kind of data we have in order to predict the WQI of lakes in Washington state. We are going to refer the article Khoi et al., (2022) for choosing the approach to perform the ML.


## ML Model Construction:

To construct the ML model for prediction of water quality index, first we need to select the proper input variables which has sufficient underlying information to predict WQI. Selection of appropriate input factors is very important as it could improve the model accuracy by eliminating the factors with undesirable impact on the predictive performance. 

For ANN, first plot the data to check if there is any outlier.

Throughout this implementation walkthrough, take some time to critically think about the following:

•	What about this dataset makes it complex? Is it a variable? Is it the distribution of values? Is it the size of the dataset?

•	Which variables should I investigate prior to implementing my model? What does the distribution look like? Hint: Use Pandas' Series.plot.density() method to find out.

•	What outcome am I looking for from the model? Which activation function should I use to get my desired outcome?

•	What is my accuracy cutoff? In other words, what percent testing accuracy must my model exceed?

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

6.	I checked the null values again and summary of the null values in each column is as follows:

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

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(19).png?raw=true)

3.	I visualized the counts in WQI using: WQI_df["WQI"].plot.density()

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(20).png?raw=true)

4.	So, in this dataset there is one dependent target variable i.e., “WQI” and 10 independent features. All of these 10 input variable feature do not have same amount of impact on the WQI and hence, to find out the extent of correlation of the input variable we are going to perform multiple correlation analysis. 


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(21).png?raw=true)


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(22).png?raw=true)


I plotted the scatter plot for each feature with the target variable to check the relationship between the two. It is the visualization way to check if there is any relationship pattern between the two variables. The "WQI" is on the x-axis and the features are on the y-axis. 


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(23).png?raw=true)


During the statistical exploratory analysis I faced the difficulty in plotting the scatter plots of all the features with the target variable in a grid format, but with the help of TA I was able to sort that out.

## C.	Machine Learning for WQI prediction

In this section of the project, I applied the neural networks to predict the WQI. 

I used the following resources:

•	Scikit-Learn

•	Pandas

•	TensorFlow

•	Keras

1.	## Data Preprocessing

I dropped the first column “CollectDate” from  the data frame as the machine learning models accepts only numerical values. Then, I assigned the target and features arrays to its respective variables and performed the splitting of the preprocessed data in to training and testing datasets.

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(24).png?raw=true)

 In the development of ANN model: 
a) while assigning the values to the x, and y variable, the model code run was giving error that the "Expected 2D array, got 1D array instead. I resolved it with mentioning the ".reshape(-1,1).



## Compile, Train and Evaluate the Model

•	I defined the input features and hidden nodes for each hidden layers for deep neural network model. 

•	I defined checkpoint path and the filenames.

•	Complied the model with loss as binary_crossentropy and optimizer as “adam”.

•	Created a callback which saves the model’s weight every 5 epochs.

•	Then evaluated the model using test data and exported to HDF5 file so that anyone can use it later.


![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(25).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(26).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(27).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(28).png?raw=true)

![Picture_1](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ritu_Gothwal/Ritu_Gothwal/ScreeenShots/Screenshot%20(29).png?raw=true)

![Picture_1]()
