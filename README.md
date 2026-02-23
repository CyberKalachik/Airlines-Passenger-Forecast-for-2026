# Airline Passenger Forecast for 2026

## Project Overview
This project analyses historical U.S. air traffic passenger data to forecast airline performance for 2026 using machine learning techniques.

Using monthly passenger statistics, the analysis estimates future demand, ranks airlines by growth, and evaluates projected market share among major carriers.
________________________________________
## Dataset
File: Air_Traffic_Passenger_Statistics.csv

Source: https://catalog.data.gov/dataset/air-traffic-passenger-statistics

The dataset contains monthly passenger counts by airline and route type.
________________________________________

## Objectives
•	Build a clean time-series dataset at the airline level
•	Forecast passenger volume for 2026
•	Calculate projected year-over-year growth
•	Identify growth leaders and market structure
•	Visualize forecast results
________________________________________

## Data Preparation
•	Merged operating and published airline names into a unified field
•	Converted dates to proper datetime format
•	Aggregated passenger counts by month and airline
•	Filtered the analysis period (2000–2025)
•	Selected the top airlines by total traffic
________________________________________

## 🤖 Forecasting Approach
A machine learning model was used for time-series forecasting.
Model: Random Forest Regressor
Feature engineering included:
•	Lag features (1 month, 12 months)
•	Rolling averages (3-month, 12-month)
•	Seasonal indicators (month, year)
Forecasts were generated for all months of 2026 and aggregated annually.
________________________________________

## Key Results
•	Fastest projected growth: Southwest Airlines (~37% YoY)
•	Largest projected airline: United Airlines (~25.1M passengers)
•	No major airlines projected to decline in 2026
________________________________________

## Visualizations
The project includes:
•	2026 passenger forecast by airline
•	YoY growth ranking with performance tiers
•	Projected market share distribution
________________________________________

## Tech Stack
•	Python
•	Pandas & NumPy
•	Scikit-learn
•	Matplotlib & Seaborn
________________________________________

## Use Cases
Demand forecasting is critical for airline operations, including:
•	Capacity planning
•	Route optimization
•	Pricing strategy
•	Workforce planning
•	Long-term strategic decisions
