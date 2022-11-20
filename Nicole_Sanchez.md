# Final_Project_UCB_Bootcamp

## Summary - Water Qality Predictor - Nicole Sanchez - The Manager

Group 9 worked on a project to predict the Water Quality Index (WQI) of 3 different lakes in King County,
Washington. Water quality is an issue that affects people globally. Specific contaminants are measured in the
water source. Those values are plugged into a formula and water quality index is calculated. The higher the
number, the better the water quality. Two models were chosen for this project an Artificial Neural Network (ANN)
model and a SKForecast model. The ANN model performed well, but did not provide a future prediction of the WQI.
The SKForecast model was then used to forecast a future WQI. A webpage was created to host the forecasted WQI
for the next 12 months. 

## Methods

### Segment 1

- Downloaded one year of raw data from https://green2.kingcounty.gov/lakes/Query.aspx
- Used MS-Excel to explore the data. 
- Created Jupyter Notebook file.
- Encountered an error while attempting to load the .csv file into the ipynb.
- Added the "encoding='unicode_escape'" parameter to the pd.read_csv function.
- Used suggestion from Ritu to check the data types, and change the "CollectDate" column to a date fromat.
- Used suggestion from Ritu to remove various unneeded columns.
- As a team, we decided to use Ritu's ipynb file to load the data, clean the data and start experimenting with different ML Models.
- Ritu was able to find a formula to calculate the Water Quality Index (WQI).
- I used that formula and was able to translate it into code that calculated the WQI for each row of
  data, and saved the value in a new column.
  
<img width="754" alt="WQI_formula_code" src="https://user-images.githubusercontent.com/106849689/202930826-8b2fe294-326d-4fdf-8fd2-6f99fd4865ff.png">

- I used that new data set to experiment with the linear regression model.
-
<img width="368" alt="Multi_Lin_Reg" src="https://user-images.githubusercontent.com/106849689/202930849-36c81d75-bc1c-411d-90a2-1350e53c9c62.png">

- I also scaled the data, then ran the linear regreassion model.

<img width="392" alt="Scaled_Multi_Lin_Reg" src="https://user-images.githubusercontent.com/106849689/202930878-819b5610-261e-40b7-b40c-a911e2b89caa.png">

- I attempted to plot the data as a scatter plot.
- I also attempted to plot the data as a bar chart.
- I created a draft of our presentation:
    https://docs.google.com/presentation/d/16ZrQ_KEqKrhyH9IsrkqKFNsgMb9nEmqp7MNBwoZUki4/edit?usp=sharing

- The presenation contained:
    - the reason why they selected this topic.
    - the description of the source of data.
    - the questions we hope to answer with the data.

### Segment 2

- I updated the presentation to add:
    - a description of how we, as a team, explored the data.
    - a desctiption of how we, as a team, analyzed the data.
- I created a simple website that:
    - displayed our logo.
    - contained a drop-down list of the lakes we are using to calculate the water quality index.
    - displayed a map image of the lake selected from the drop-down list.

<img width="898" alt="WQI_Pred_Webpage" src="https://user-images.githubusercontent.com/106849689/202930929-b3819be8-3dde-4c8b-be72-c2655f5a8bfc.png">


### Segment 3

- I worked with the team to update the presentation with their own visuals for the slides they are
  presenting.
- We, as a team, created slides that showcased the technologies, languages, tools, and algorithms
  used throughout the project
- I practiced how I would present my slides during our group presentation.
- I updated the website by:
    - adding more formatting to the text and background.
    - replacing the map images with interactive maps.

### Segment 4

- I updated the presentation to:
    - include the result of analysis.
    - add recommendation for future analysis.
    - describe anything the team would have done differently.
- I updated the website to include the WQI forecasts for the last 2 lakes.

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

