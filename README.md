# MovieLens 20M Dataset Analysis & Movie Recommendations

This project provides an interactive analysis and recommendation system using the MovieLens 20M dataset. It covers data exploration, preprocessing, association rule mining, and movie recommendations based on genre patterns. The application is built with **Streamlit**, **Pandas**, **Plotly**, and **MLxtend**.

## Table of Contents

1. Overview
2. Features
3. Technologies
4. Running the Application
5. Project Structure
6. Acknowledgments
7. License

## Overview

The **MovieLens 20M dataset** includes millions of ratings, movie metadata, and user-generated tags. This project enables users to:

- Explore movies, ratings, links, and tags
- Preprocess data (handle missing values, convert timestamps, encode genres)
- Visualize insights from exploratory data analysis
- Apply the Apriori algorithm for association rule mining on genres
- Generate recommendations based on genre association rules

Interactive controls allow adjustment of support, confidence, and other parameters with real-time visualization.

## Features

- **Data Overview**: Dataset statistics and sampling
- **Data Preprocessing**: Missing data handling, timestamp conversion, genre encoding
- **Exploratory Data Analysis**: Top-rated movies, rating distributions, temporal trends, genre breakdowns
- **Association Rule Mining**: Frequent genre itemsets, rule generation, visualization
- **Movie Recommendations**: Suggest movies based on discovered genre relationships
- **Interactive Visualizations**: Dynamic charts and tables

## Technologies

- Python (**Streamlit**, **Pandas**, **Plotly**, **MLxtend**, **NumPy**)
- Apriori algorithm for association rule mining
- Interactive dashboards and visualizations

## Running the Application

```bash
git clone https://github.com/your-username/final-project-data-science.git
cd final-project-data-science

python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

## Project Structure

```
final-project-data-science/
│
├── data/                   # Directory for storing data files
│   ├── ml-20m.zip          # Compressed MovieLens 20M dataset
│   ├── ml-20m.zip.md5      # MD5 checksum for verifying dataset integrity
│   ├── genome-scores.csv   # Tag genome scores
│   ├── genome-tags.csv     # Tag genome tags
│   ├── links.csv           # Movie links to external databases
│   ├── movies.csv          # Movie metadata
│   ├── ratings.csv         # User ratings for movies
│   └── tags.csv            # User-generated tags for movies
│
├── notebooks/              # Jupyter notebooks for exploration and analysis
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_eda_visualization.ipynb
│   ├── 04_association_rule_mining.ipynb
│   └── 05_movie_recommendations.ipynb
│
├── src/                    # Source code for the application
│   ├── __init__.py
│   ├── app.py              # Main application file
│   ├── data_processing.py   # Data loading and preprocessing functions
│   ├── eda.py              # Exploratory data analysis functions
│   ├── model.py            # Association rule mining and recommendation functions
│   └── visualization.py     # Visualization functions
│
├── requirements.txt         # Python package dependencies
├── README.md                # Project documentation
└── .gitignore               # Git ignore file
```

## Acknowledgments

- [MovieLens](http://movielens.org) for providing the dataset
- [GroupLens Research](http://grouplens.org) for their support and tools
- The open-source community for their valuable libraries and frameworks

## License
This project is not licensed for public use. All rights reserved.
