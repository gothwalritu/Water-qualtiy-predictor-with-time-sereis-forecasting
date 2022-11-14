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

### Segment 1 ###

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
     WQ0611_df = WQ0611_df.drop(columns=["Depth (m)", "Ammmonia Nitrogen Qualifier*", "Cond Qualifier*", "DO Qualifier*", "Ecoli Qualifier*", "Fecal   
     Coliform Qualifier*","Nitrate Nitrite Qualifier*", "OP Qualifier*", "pH Qualifier*", "Temperature Qualifier*", "TN Qualifier*","TP Qualifier*",            "Total Alkalinity Qualifier*"])

  # Drop all the MDL columns
    WQ0611_df = WQ0611_df.drop(columns=["AN MDL (mg/L)", "Cond MDL (µmhos/cm)", "DO MDL (mg/L)", "Nitrate Nitrite MDL (mg/L)", "OP MDL (mg/L)", "TN MDL       (mg/L)", "TP MDL (mg/L)", "Total Alkalinity MDL (mg/L)"])
  
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

### Segment 2 ###

Goal:
-	Applied 2 machine learning models to the entire lake dataset 
-	See if logistic regression model is working
-	See if random forest model is working

Tasks:
-	Applied logistic regression model on 1 testing site 0611
-	Applied random forest model on 1 testing site 0611 

__Logistic Regression Model for 0611 testing site__

```
    # Seperate the Features (X) from the Target (y)
    y = df_3['WQ']
    X = df_3.drop(['WQ', 'CollectDate'], axis='columns')

    # Split our data into training and testing 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)
    X_train.shape

    # Create a Logistic Regression Model
    classifier = LogisticRegression(solver='lbfgs', random_state=0)
    classifier
    classifier.fit(X_train, y_train)
  
```
Problem encounter: data was continuous. The solution was to convert the data in order to able to run the model. 

```
    # Import model for convert data
    from sklearn import preprocessing
    from sklearn import utils

    # Convert y values to categorical values
    lab = preprocessing.LabelEncoder()
    y_transformed = lab.fit_transform(y)

    # Prediction outcomes for test data set 
    predictions = classifier.predict(X_test)
    results = pd.DataFrame({"Prediction": predictions, "Actual": y_test}).reset_index(drop=True)
    results.head(20)

    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test, predictions))

```

Result:
![0611_preditction_actual](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Ling_Hoang/Code%20and%20Visual%20Snippets/0611_Prediction_Actual.png)

Accuracy score is 0.0

__Random Forest Model for 0611 testing site__

```
    # Define the features set
    X = WQ0611_clean_df.copy()
    X = X.drop(['WQ','CollectDate'], axis='columns')
    X.head()

    # Define the target set.
    y = WQ0611_clean_df["WQ"].ravel()
    y[:20]

    # Splitting into Train and Test sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=78)

    # Creating a StandardScaler instance.
    scaler = StandardScaler()
    # Fitting the Standard Scaler with the training data.
    X_scaler = scaler.fit(X_train)

    # Scaling the data.
    X_train_scaled = X_scaler.transform(X_train)
    X_test_scaled = X_scaler.transform(X_test)

    # Fit the random forest classifier
    # Create a random forest classifier.
    rf_model = RandomForestClassifier(n_estimators=128, random_state=78)

    # Fitting the model
    rf_model = rf_model.fit(X_train_scaled, y_train)

    # Making predictions using the testing data.
    predictions = rf_model.predict(X_test_scaled)

    # Calculating the confusion matrix.
    cm = confusion_matrix(y_test, predictions)

    # Create a DataFrame from the confusion matrix.
    cm_df = pd.DataFrame(
        cm, index=["Actual 0", "Actual 1", "Actual 2", "Actual 3", "Actual 4", "Actual 5"],
        columns=["Predicted 0", "Predicted 1", "Predicted 2", "Predicted 3", "Predicted 4", "Predicted 5"])
    cm_df


```

![0611_Prediction_actual]
(https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Ling_Hoang/Code%20and%20Visual%20Snippets/0611_RF_ConfusionMatrix.png)

```
    # Calculating the accuracy score.
    acc_score = accuracy_score(y_test, predictions)# Displaying results
    print("Confusion Matrix")
    display(cm_df)
    print(f"Accuracy Score : {acc_score}")
    print("Classification Report")
    print(classification_report(y_test, predictions))
    
    # Calculate feature importance in the Random Forest model.
    importances = rf_model.feature_importances_
    importances

    # We can sort the features by their importance.
    sorted(zip(rf_model.feature_importances_, X.columns), reverse=True)

```
![0611_importances](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Ling_Hoang/Code%20and%20Visual%20Snippets/0611_RF_importances.png)

__ Set Up Machine Learning Model on Lake Sammamish__

During Segment 1, I only applied both models on to one part of the lake. Now when all the data sites have been appended, I will apply both models on the merged dataset. 

 __Logistic Regression Model for Lake Sammamish__
 
 | Prediction/Actual Model on 0611 testing site | Preditction/Actual Model on Big Lake | 
| :---:         |     :---:      |  
|![0611 testing site](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Ling_Hoang/Code%20and%20Visual%20Snippets/0611_Prediction_Actual.png)  |![Big_Lake](https://github.com/gothwalritu/Final_Project_UCB_Bootcamp/blob/Ling_Hoang/Ling_Hoang/Code%20and%20Visual%20Snippets/BL_Prediction_Actual.png)   |

__Random Forest Model for Lake Sammamish__









