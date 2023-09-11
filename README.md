# Laptops-prices-regression-end-to-end

### Problem Definition
Predict, based on many laptop features, a laptop price.
### Data
This dataset is a collection of features related to various laptops, such as brand, processor type, RAM, storage capacity, and other specifications. The dataset also includes the corresponding prices of these laptops. This dataset can be used for regression analysis to predict the prices of laptops based on their features. The dataset is suitable for data scientists, machine learning enthusiasts, and researchers who are interested in building regression models to predict the prices of laptops based on various features.

The dataset used can be found on Kaggle using the following link: https://www.kaggle.com/datasets/anubhavgoyal10/laptop-prices-dataset

### Goal

> We aim to achieve an RMSE value as close to zero as possible.

### Evaluation Metric
For the proper evaluation we gonna calculate:
* Root Mean Squared Error (RMSE)
* R-squared (R²) Score
* Mean Absolute Error (MAE)

By using them we'll gain insights into different aspects of model predictions.

### Features
##### This is our data dictionary
brand: the laptop's brand [ASUS, DELL, Lenovo, HP, acer]
processor_brand: processor's brand [Intel, AMD, M1]
processor_name: name of the processor [Core i3, Core i5, Core i7, Ryzen 5, Ryzen 7]
processor_gnrtn: the processor generation [Not Available, 7th, 8th, 10th, 11th]
ram_gb: ram size [4 GB, 8 GB, 16 GB, 32 GB]
ram_type: ram type [DDR4, DDR5, LPDDR3, LPDDR4, LPDDR4X]
ssd: Solid-State Drive -> has or not ssd and if does, storage capacity [0 GB, 128 GB, 256 GB, 512 GB, 1024 GB]
hdd: Hard Disk Drive -> has or not ssd and if does, storage capacity [0 GB, 512 GB, 1024 GB, 2048 GB]
os: Operating System [Windows, Mac, DOS]
os_bit: Operating System bit [32-bit, 62-bit]
graphic_card_gb: graphic card size [0 GB, 2 GB, 4 GB, 6 GB, 8 GB]
weight: laptop's weight [Casual, ThinNlight, Gaming]
warranty: has or not and if does, how much time [No warranty, 1 year, 2 years, 3 years]
Touchscreen: has or not [No, Yes]
msoffice: MS office present or not [No, Yes]
Price: Price (in indian rupee)
rating: laptop's rating in stars [1 star, 2 stars, 3 stars, 4 stars, 5 stars]
Number of Ratings: number of ratings
Number of Reviews: number of reviews

### Tools
We're going to use pandas, Matplotlib and NumPy for data analysis and manipulation. We'll work with the following algorithms:
* Random Forest Regressor
* Linear Regression
* XGBoost
* SVR
