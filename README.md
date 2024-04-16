# dsa4212_lstm

## Introduction

This repository contains a Long Short-Term Memory (LSTM) model implemented from scratch using mainly `numpy` and `jax` Python libraries. 

The purpose of this LSTM model is to perform prediction on taxi demand time series data. Specifically, we trained and evaluated the model on yellow taxi trip records from Manhattan spanning the years 2022 and 2023 in the [TLC Trip Record dataset](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). 

## Repository structure: 

- [run.ipynb](/run.ipynb): Full walkthough of data preparation, model training, and evaluation using MSE and line plot.
- [Taxi_data_processing.ipynb](/Taxi_data_processing.ipynb): Data preprocessing code.
- [model/](/model/): LSTM model and helper functions.
