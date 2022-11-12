# Final_Project_UCB_Bootcamp

## Segment 1 Activities

- Downloaded one year of raw data from https://green2.kingcounty.gov/lakes/Query.aspx
- Used MS-Excel to explore the data. 
- Created Jupyter Notebook file.
- Encountered an error while attempting to load the .csv file into the ipynb.
- Added the "encoding='unicode_escape'" parameter to the pd.read_csv function.
- Used suggestion from Ritu to check the data types, and change the "CollectDate" column to a date fromat.
- Used suggestion from Ritu to remove various unneeded columns.
- As a team, we decided to use Ritu's ipynb file to load the data, clean the data and start experimenting with different ML Models.
- Ritu was able to find a formula to calculate the Water Quality Index (WQI).
- I used that formula and was able to translate it into code that calculated the WQI for each row of data, and saved the value in a new column.
- I used that new data set to experiment with the linear regression model.
- I also scaled the data, then ran the linear regreassion model.
- I attempted to plot the data as a scatter plot.
- I also attempted to plot the data as a bar chart.
- I created a draft of our presentation.

## Segment 2 Activities

- I updated the draft of our presentation with the segment 2 topics
- I created a simple website that:
    - displays our logo.
    - contains a drop-down list of the lakes we are using to calculate the water quality index.
    - displays a map of the lake selected from the drop-down list.