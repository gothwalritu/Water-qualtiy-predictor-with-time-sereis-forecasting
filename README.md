# Final_Project_UCB_Bootcamp

## Introduction

We all love recreation activities in and around water. We always wonder where we should go, and which locations have the safest water for
recreational activities, as well as which locations have potable water.

In this project we will develop a tool which will show locations around Lake Sammamish in Washington State, and display which are safe for
recreational and potability use. It will be an interactive tool which takes user input for geolocation and recommends the locations which 
are fit for potability and recreational uses. 

## ERD

We used Quick DBD to create a ERD of our data
![QuickDBD-export](https://user-images.githubusercontent.com/106849689/200139417-deccad9a-fb82-4ff2-92a1-862dc4d4b325.png)

## Source citation for raw data

https://green2.kingcounty.gov/lakes/Query.aspx

## ML models 

We wanted to predict the water quality index (WQI) as a target variable which is dependent on independent feature variables such as using temperature, pH, turbidity, dissolved oxygen (DO), biochemical oxygen demand (BOD), and concentrations of other pollutants). Once, the WQI is known we wanted to predict it for future years. So, we  to applied two machine learning models:

A) Supervised regression ANN model.
B) Time series forecasting model.

## Link to Google Slides draft presentation:

https://docs.google.com/presentation/d/16ZrQ_KEqKrhyH9IsrkqKFNsgMb9nEmqp7MNBwoZUki4/edit#slide=id.g18d6e3615c7_5_1
