import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # load messages & categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # Merge the messages and categories datasets using the common id & Assign this combined dataset to df 
    df = messages.merge(categories, on='id')         
    return df


def clean_data(df):
    # Split the values in the categories column on the ; character so that each value becomes a separate column.
    disaster_categories = df['categories'].str.split(';', n=36, expand=True)
    
    # Using the first row of categories dataframe to create column names for the categories data.
    first_row = disaster_categories.iloc[0]
    
    # The last 2 charachters are sliced so that only the name of the column remains
    category_colnames = first_row.map(lambda x: x[:-2])
    
    # rename the columns of `categories`
    disaster_categories.columns = category_colnames
    
    # Iterate through the category columns in df to keep only the last character of each string. For example, related-0 becomes 0, related-1 becomes 1
    for column in disaster_categories:
        # set each value to be the last character of the string
        disaster_categories[column] = disaster_categories[column].map(lambda x: x[-1:])

        # convert column from string to numeric
        disaster_categories[column] = disaster_categories[column].astype(int)
    
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,disaster_categories], axis=1)
    
    # drop duplicates based on the columns 'message' and 'genre'
    df = df.drop_duplicates(subset=['message','genre'])
    
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///database_filename.db')
    return df.to_sql('database_filename', engine, index=False)
      


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