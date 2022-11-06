# Water Quality Predictor # 

### Goal ###

To predict the water quality of a specific body of water for recreational use and potability. And using Machine Learning Models for predicting the water quality index in Lake Sammamish which is 8 miles east of Seattle in King County, Washington, United States.

Our Initial hypothesises: 
  - Is the water safe for recreational activities?
  - Is the water safe to drink?
  - What is the water temperature?

### Resources ### 

We used [King County](https://green2.kingcounty.gov/lakes/Query.aspx) website to pull all the database for Lake Sammamish. This included 7 sites with 14 water parameters. 

We also read this article on [Using Machine Learning Models for Predicting the Water Quality Index in the La Buong River, Vietnam](https://www.mdpi.com/2073-4441/14/10/1552) to calculate the water quality index for Lake Sammamish, and how this case study using different machine learning models to apply on our dataset. 

### Database ###

![Database](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Erica_Gutierrez/QuickDBD-export.png)

There are 7 test sites at Lake Sammamish and we will analyze and combine all the test site to run the machine learning models. 

### Segment 1 Findings ###

Goal: to familiar with the data. We will analyze the first testing site and apply Ml model first. 

```
  Import and Sanitize the 1st set of Lake Sammamish Data = WQ0611
  
  # Load the file data into the path
  file_path = "../GroupProject/Resources/Lake_Sammamish_0611.csv"

  # Load the file_path data into dataframe
  WQ0611_df = pd.read_csv(file_path, skiprows=12, encoding='unicode_escape', on_bad_lines='skip')

  # Show the WQ dataframe 
  WQ0611_df.head(10)
```
Since the csv file from King County contains extra infomations on the top of the file, we have to import it without the first 12 rows. The import results in 10 rows and 36 columns. 

```
# Drop all the qualifiers 
   WQ0611_df = WQ0611_df.drop(columns=["Depth (m)", "Ammmonia Nitrogen Qualifier*", "Cond Qualifier*", "DO        Qualifier*", "Ecoli Qualifier*", "Fecal Coliform Qualifier*","Nitrate Nitrite Qualifier*", "OP            Qualifier*", "pH Qualifier*", "Temperature Qualifier*", "TN Qualifier*","TP Qualifier*", "Total            Alkalinity Qualifier*"])

# Drop all the MDL columns
  WQ0611_df = WQ0611_df.drop(columns=["AN MDL (mg/L)", "Cond MDL (µmhos/cm)", "DO MDL (mg/L)", "Nitrate         Nitrite MDL (mg/L)", "OP MDL (mg/L)", "TN MDL (mg/L)", "TP MDL (mg/L)", "Total Alkalinity MDL             (mg/L)"])
```
After the first look at overall data, we decided to drop all qualifiers columns and all the MDL columns since it doesn't have the importance in calculating the water quality index. 

```
  WQ0611_df = WQ0611_df.drop(columns=["Unnamed: 35", "Ecoli", "Total Alkalinity (mg/L)"])
```
We also dropped those extra columns since it not needed. This dropped results in 10033 rows and 12 columns. 

```
  # Convert string to datetime
    WQ0611_df["CollectDate"] = pd.to_datetime(WQ0611_df["CollectDate"])
  
  # Apply lambda function to fill NaN space. 
    df_2 = df_1.groupby(["CollectDate"]).apply(lambda x:x.fillna(x.mean()))
  
  # Drop all NaN column
    df_3 = df_2.dropna()
    
  # Check the number of columns after dropping all Nan values
    df_3.count()
```
This result in total of 3015 rows and 11 columns.
