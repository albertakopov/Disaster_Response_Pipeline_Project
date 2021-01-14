import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import pickle
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sqlalchemy import create_engine

from sklearn.metrics import confusion_matrix,classification_report, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///../data/DisasterResponse.db')

    # load to database
    df = pd.read_sql_table('DisasterResponse' ,engine) 

    # define features and label arrays
    X = df['message']
    y = df.iloc[:, 4:]
    
    # Extract the column names 
    category_names = list(df.columns[4:])
    return X, y, category_names
    
def tokenize(text):
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex,text)
    
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url,"urlplaceholder")

    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()


    # lemmatize, normalize case, and remove leading/trailing white space    
    clean_tokens = [lemmatizer.lemmatize(w).lower().strip() for w in tokens]

    return clean_tokens


def build_model():
    # text processing and model pipeline
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('text_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),
    ])),

    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    # define parameters for GridSearchCV
    parameters = {
    'clf__estimator__learning_rate':[0.01],
    }

    # create gridsearch object and return as final model pipeline
    model_pipeline = GridSearchCV(estimator=pipeline, param_grid=parameters)


    return model_pipeline



def evaluate_model(model, X_test, y_test, category_names):
    
    #Determine predicted values
    y_pred = model.predict(X_test)
    
    #Calculate the accuracy, precision, and recall of the tuned model
    print(classification_report(y_test.iloc[:, 1:].values, np.array([x[1:] for x in y_pred]), target_names = category_names))
    print("\nBest Parameters:", model.best_params_)


def save_model(model):
    # save the model to disk
    pickle.dump(model, open('model.pkl', 'wb'))
    


def main():
    #if len(sys.argv) == 3:
        database_filepath, model_filepath = ['DisasterResponse.db','model.pkl']#sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model)

        print('Trained model saved!')

    #else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()