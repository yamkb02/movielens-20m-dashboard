# MovieLens 20M Dataset Analysis & Movie Recommendations

This project offers an interactive analysis of the MovieLens 20M dataset, focusing on data exploration, preprocessing, association rule mining, and generating movie recommendations based on genre patterns. Built with **Streamlit**, **Pandas**, **Plotly**, and **MLxtend**, it provides an intuitive UI and dynamic visualizations.

## Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Technologies](#technologies)
4. [Running the Application](#running-the-application)
5. [Project Structure](#project-structure)
6. [Acknowledgments](#acknowledgments)

## Overview

The **MovieLens 20M dataset** contains rich information on movie ratings, genres, and tags enabling comprehensive analysis and recommendation systems. This project lets users:

* Explore the Movies, Ratings, and Tags datasets
* Preprocess data (handle missing values, convert timestamps)
* Visualize key insights from exploratory data analysis
* Apply **Apriori algorithm** for association rule mining on movie genres
* Generate recommendations based on genre association rules

Interactive controls allow adjustment of support, confidence, and other parameters with real-time visualization.

## Features

* **Data Overview**: Dataset statistics and sampling options
* **Data Preprocessing**: Missing data handling, timestamp conversion, genre encoding
* **Exploratory Data Analysis**: Top-rated movies, rating distributions, temporal trends, genre breakdowns
* **Association Rule Mining**: Frequent genre itemsets, rule generation, and visualization
* **Movie Recommendations**: Suggest movies based on discovered genre relationships
* **Interactive Visualizations**: Dynamic charts and tables powered by Plotly and Streamlit

## Technologies

* Python (**Streamlit**, **Pandas**, **Plotly**, **MLxtend**, **NumPy**)
* Association Rule Mining with Apriori algorithm
* Data visualization and interactive dashboards

## Running the Application

```bash
git clone https://github.com/your-username/movielens-recommendation.git
cd movielens-recommendation

python3 -m venv env
source env/bin/activate  # Windows: env\Scripts\activate
pip install -r requirements.txt

streamlit run app.py
```

## Project Structure

```
movielens-recommendation/
├── app.py                 # Streamlit app
├── requirements.txt       # Dependencies
├── data/                  # Dataset files (movies.csv, ratings.csv, tags.csv)
├── assets/                # Images and animations
│   └── lottie_closing.json
├── README.md
└── LICENSE
```

## Acknowledgments

* **GroupLens Research**, University of Minnesota — MovieLens 20M dataset
* **Streamlit** — interactive web apps
* **MLxtend** — Apriori implementation
* **Plotly** — interactive visualizations

## License

MIT License — see [LICENSE](LICENSE)
