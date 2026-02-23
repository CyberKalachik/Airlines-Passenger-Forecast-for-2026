# Airline Passenger Forecast for 2026

## Project Overview
This project analyses historical air traffic passenger data of San Francisco International Airport to forecast airline performance for 2026 using machine learning techniques.

Using monthly passenger statistics, the analysis estimates future demand, ranks airlines by growth, and evaluates projected market share among major carriers.
________________________________________
## Dataset
File: Air_Traffic_Passenger_Statistics.csv

Source: https://catalog.data.gov/dataset/air-traffic-passenger-statistics

The dataset contains monthly passenger counts by airline and route type.
________________________________________

## Objectives
- [x] Build a clean time-series dataset at the airline level
- [x] Forecast passenger volume for 2026
- [x] Calculate projected year-over-year growth
- [x] Identify growth leaders and market structure
- [x] Visualize forecast results
________________________________________

## Data Preparation
- Merged operating and published airline names into a unified field
- Converted dates to proper datetime format
- Aggregated passenger counts by month and airline
- Filtered the analysis period (2000–2025)
- Selected the top airlines by total traffic

``` Python

import pandas as pd
import numpy as np

# Load the raw air traffic passenger dataset from CSV
df = pd.read_csv(
    r'C:\Users\bobma\PycharmProjects\ML\Air_Traffic_Passenger_Statistics.csv'
)

# ✅ Convert date column BEFORE any grouping
df['Activity Period Start Date'] = pd.to_datetime(
    df['Activity Period Start Date'],
    errors='coerce'
)

# Create a unified airline column:
# use the operating airline when available,
# otherwise fall back to the published airline
df['Airline'] = df['Operating Airline'].fillna(df['Published Airline'])

# (optional but recommended) drop rows with missing airline or date
df = df.dropna(subset=['Airline', 'Activity Period Start Date'])

# Aggregate passenger counts by month and airline
# This produces total passengers transported by each airline per month
airline_monthly = (
    df.groupby(['Activity Period Start Date','Airline'])['Passenger Count']
      .sum()
      .reset_index()
      .sort_values('Activity Period Start Date')
)
```
________________________________________

## 🤖 Forecasting Approach
A machine learning model was used for time-series forecasting.
Model: Random Forest Regressor
Feature engineering included:
- Lag features (1 month, 12 months)
- Rolling averages (3-month, 12-month)
- Seasonal indicators (month, year)
Forecasts were generated for all months of 2026 and aggregated annually.
________________________________________

## Key Results
- [x]	Fastest projected growth: Southwest Airlines (~37% YoY)
- [x]	Largest projected airline: United Airlines (~25.1M passengers)
- [x]	No major airlines projected to decline in 2026
________________________________________

## Visualizations
The project includes:
- [x]	2026 passenger forecast by airline
- [x]	YoY growth ranking with performance tiers
- [x]	Projected market share distribution
________________________________________

## Tech Stack
- [x]	Python
- [x]	Pandas & NumPy
- [x]	Scikit-learn
- [x]	Matplotlib & Seaborn
________________________________________

## Use Cases
Demand forecasting is critical for airline operations, including:
- [x]	Capacity planning
- [x]	Route optimization
- [x]	Pricing strategy
- [x]	Workforce planning
- [x]	Long-term strategic decisions
