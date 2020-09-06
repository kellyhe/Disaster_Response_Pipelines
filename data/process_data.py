# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """ load and merge messages dataset and categories dataset
        Inputs: messages data filepath, categories data filepath
        Output: merged dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on ="id",how='inner')
    return df


def clean_data(df):
    """Clean dataset including:
       1. Drop duplicates;
       2. Split categories into separate category columns. (36 target measures)
       Input: merged data frame
       Output: cleaned data frame
    """
    # Drop duplicates
    df = df.drop_duplicates()
    # Split categories into separate category columns
    categories2 = df.categories.str.split(";", expand=True)
    # Assign a list of new column names
    category_colnames = categories2.iloc[0].str[:-2].tolist()
    categories2.columns = category_colnames
    # Convert category values to just numbers 0 or 1.
    for column in categories2:
        categories2[column] = categories2[column].str[-1:].astype(int)
    # Replace categories column in df with new category columns.
    df = pd.concat([df, categories2], axis=1)
    # drop the original categories column
    df.drop('categories',axis =1, inplace =True)
    return df

def save_data(df, database_filename):
    """Save the clean dataset into an sqlite database.
       Inputs: data, database filename
       Output: saved data into an sqlite database
    """
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()