# Water Quality Prediction 

To predict the water quality of a specific body of water for recreational use and potability. A water quality index will be used to determine the level of quality.


## Roadmap for Database:
- Download the raw data from https://green2.kingcounty.gov/lakes/Query.aspx
- Once the raw data was downloaded from the Lake Services webiste, the data was cleaned, mergered into one final dataset.
- From here were we able to create tables from the results of the ML model.

### Create QuickDBD 
<img width="1581" alt="QuickDBD" src="https://user-images.githubusercontent.com/107595578/204201501-aa4f3ca1-48a6-452b-8598-ed76a2b73982.png">

### Use Big Lake Sammamish data to create Database for Lake Sammamish
<img width="1007" alt="" src="https://user-images.githubusercontent.com/107595578/204202243-678c3166-7d8c-4558-9e97-2a637d08ffa4.png">

### Use SQLite for Database 
 
<img width="400" alt="" src="https://user-images.githubusercontent.com/107595578/204201089-acf3a45c-a595-43e4-95ab-61e2a17e7f01.png">

### Make a connection to the SQL Database

<img width="400" alt="Screen Shot 2022-11-27 at 9 29 57 PM" src="https://user-images.githubusercontent.com/107595578/204201107-b8947477-1019-4834-a66d-b59dae9f2386.png">

### Create tables for Database 

```# WQI Predictions_Vs_Actual in the Excellent Range 
cursor=c.execute('''
SELECT Prediction,Actual
FROM Predictions_Vs_Actual
WHERE Prediction BETWEEN 91 and 100
ORDER BY Prediction;''')
for row in cursor:
    print(row)
```
 Using the predictions from the ML model we were able to filter our data into the preset ranges of water quality index. 
 Excellent (WQI =100+ TO 91), Good (WQI =90 TO 76), Fair (WQI =75 TO 51), Poor (WQI =50 TO 26), Very Poor (WQI =25 TO 0)
 
<img width="400" alt="Screen Shot 2022-11-27 at 9 29 31 PM" src="https://user-images.githubusercontent.com/107595578/204200981-1432b8fb-eb37-443e-81d9-c58b6b58f7c6.png">

<img width="450" alt="Tables for database" src="https://user-images.githubusercontent.com/107595578/204200675-a58d930e-8448-4e2c-bb40-dbde7e953513.png">

### Close the connection to the database  

<img width="326" alt="Screen Shot 2022-11-27 at 9 42 31 PM" src="https://user-images.githubusercontent.com/107595578/204202625-d38ff81c-be87-48a5-abca-17024a8ed987.png">

