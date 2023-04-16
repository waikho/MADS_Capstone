# README FOR CAPASTONE PROJECT BY KCT CAPITAL

## Introduction
**Can retail traders beat the market with the help of machine learning?**

One of the most challenging aspects in stocks trading is to take emotions out of your trading decisions. Greed, FOMO (fear of missing out), overconfidence, hesitations, panic - these are often the worst enemies in trading that result in losses.

We believe that the machine and data can be our allies in trading as they are emotionless when it comes to making trading decisions. Advances in Financial Machine Learning (“AFML”, by Prof. Marco Lopez de Prado, 2018, John Wiley & Sons, Inc.) is one of the most renowned textbooks on applying machine learning techniques to trading. 

This repository contains implementations and demonstrations of certain techniques introduced in AFML, using machine learning and reinforcement learning techniques.

In long run, we aim to build a fully automated trading bot that can generate consistent positive returns.

### Folder structure

    .
    ├── .github                 # Github related files (e.g. CI/CD pipeline yml - a.k.a. Actions in github 
    ├── app                     # The main application
    ├── docs                    # Documentation files
    ├── sample                  # Sample configuration files or sample scripts to help understand the main application
    ├── schema                  # Database Schema Files
    ├── scripts                 # Script files that is not part of the main applicaiton but neccessary to enable the pipeline (e.g. crontab)
    └── README.md

## Setup
Before running the machine learning models, the data collection pipeline has to be set up to enable daily automatic update of stock data. Please follow the steps below. It is assumed that you have set up a Virtual Machine running Ubuntu v20 or above.

### 1. Install Postgres Server
The 
### 2. Setup Postgres Schema
### 3. Register Alpaca Account
### 4. Register PubProxy Account
### 5. Setup Remote SSH key for CI/CD (Optional)
### 6. Create config.py
### 7. Install required modules
### 8. Setup crontab
Run the following command to set up daily scheduled job for the pipeline.
'''
crontab -u YOUR_USER_NAME -e
'''

In the editor, type the following and edit according to your time zone to set up a schedule job to run after stock market closing. The example below assumes Asia/Hong Kong time zone so it runs at 5am every day to start getting the latest stock market data.

'''
0 5 * * * /PATH_TO_YOUR_HOME/capstone/start.sh
'''

You may use the tool at https://crontab.guru/ to help getting the correct crontab expression.


## Usage
### 1. Meta-Labeling & the Triple Barrier Method
### 2. Reinforcement Learning

