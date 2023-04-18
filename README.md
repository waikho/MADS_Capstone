# README FOR CAPASTONE PROJECT BY KCT CAPITAL

## Introduction
One of the most challenging aspects in stocks trading is to take emotions out of your trading decisions. Greed, FOMO (fear of missing out), overconfidence, hesitations, panic - these are often the worst enemies in trading that result in losses.

In this capstone project, we have built a prototype “trading bot” using machine learning and reinforcement learning techniques. 

### Architecture Overview

![alt text](https://github.com/waikho/MADS_Capstone/blob/main/assets/architecture.png?raw=true)


### Folder Structure

    .
    ├── .github                 # Github related files (e.g. CI/CD pipeline yml - a.k.a. Actions in github)
    ├── app                     # The data pipeline application
    ├── assets                  # static assets including diagrams or other media file
    ├── sample                  # Sample configuration files or sample scripts for the data pipeline application
    ├── schema                  # Database Schema Files
    ├── ML                      # Modules and Notebook Files for Meta-Labeling & the Triple Barrier Method
    ├── RL                      # Modules and Notebook Files for Reinforcement Learning
    └── README.md

## Setting up the Data Collection Pipeline
Before running the machine learning models, the data collection pipeline has to be set up to enable daily automatic update of stock data. Please follow the steps below. It is assumed that you have set up a Virtual Machine running Ubuntu v20 or above.

### 1. Install Postgres Server
Follow this guide (https://ubuntu.com/server/docs/databases-postgresql) to install Postgres SQL server. Then set up a user account.

### 2. Setup Postgres Schema
Create a new database. Then grant all privileges under the database to the user created in step 1. Load the SQL files under the /schema folder to the database to create all necessary tables. 

### 3. Register Alpaca Account
Register an account at https://app.alpaca.markets/signup. Only paper trading account is required to run this repository and acquire non-realtime data. Once an account is created, you can visit https://app.alpaca.markets/paper/dashboard/overview and choose "View API Keys" at the right hand side of the screen to obtain your API key.
![alt text](https://github.com/waikho/MADS_Capstone/blob/main/assets/alpaca_registration.png?raw=true)

### 4. Register PubProxy Account
Visit http://pubproxy.com/ and create an API key for free. Free API usage is limited and of lower priority. You may wish to upgrade to premium API for unlimited proxy requests to ensure smooth data download.

### 5. Setup Remote SSH key for CI/CD (Optional)
### 6. Create config.py
Copy sample/config-sample.py to app/config.py and update all actual crendtials acquired from the previous steps.

### 7. Install required modules
Run
```
pip install -r requirements.txt
```
or
```
pip3 install -r requirements.txt
```
to install all required python modules.


### 8. Setup crontab
Run the following command to set up daily scheduled job for the pipeline.
```
crontab -u YOUR_USER_NAME -e
```

In the editor, type the following and edit according to your time zone to set up a schedule job to run after stock market closing. The example below assumes Asia/Hong Kong time zone so it runs at 5am every day to start getting the latest stock market data.

```
0 5 * * * /PATH_TO_YOUR_HOME/capstone/start.sh
```

Make sure that you have made start.sh executable by setting chmod to 711. You may use the tool at https://crontab.guru/ to help getting the correct crontab expression.


## Usage
### 1. Meta-Labeling & the Triple Barrier Method
#### Setting up
- Copy the db_config-sample.py to db_config.py in the ML directory.
- Create a Postgres user with read-only access to the database and update db_config.py accordingly.
- Run pip to install the requirements.txt within the directory.
#### Training the model

### 2. Reinforcement Learning
#### Setting up
- Copy the db_config-sample.py to db_config.py in the ML directory.
- Create a Postgres user with read-only access to the database and update db_config.py accordingly.
- Run pip to install the requirements.txt within the directory.
#### Training the model
