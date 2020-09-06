# Disaster Response Pipeline Project

## Intro: 
In this Project, we analyzed a data set containing messages during disaster events and created a machine learning pipeline to categorize these events. 

## File Description:
This analysis contains there components:

1. ETL pipeline saved in the file 'process_data.py' in the folder 'data' include:
   a. Loads and merges messages and categories dataset.
   b. Cleans the dataset by droping duplicates.
   c. Splits categories into separate category columns (36 target measures).
   d. Saves the clean dataset into an sqlite database.
   
2. Machine Learning pipeline saved in the file 'train_classifier.py' in the folder 'models' include:
   a. Loads the clean dataset from an sqlite database.
   b. Splits the dataset into training and test sets.
   c. Builds a text processing and multiple output Random Forests machine learning pipeline.
   d. Trains and tunes the pipeline using GridSearchCV.
   e. Exports final model as a pickle file.
   
3. A web app saved in the folder 'app' include:
   a. Taking the new message and classify them into 36 categories.
   b. Visulizations of the data include: 
      1). Distribution of Message Genres
      2). Average Message Length by Genres
      3). Top 10 Categories by Message Counts

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/model.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Results:
Overall Average Accuracy for final model is 0.987684337656.

## Data Source:
Disaster data is provided from Figure Eight.