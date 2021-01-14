### Disaster Response Pipeline Project

* **Data Processing**:

* **Machine Learning Pipeline**:

* **Web Application**:

### Software & Libaries

The following python packages are used:
* [Pandas](https://pandas.pydata.org)
* [Numpy](https://numpy.org)
* [scikit-learn](https://scikit-learn.org/stable/)
* [sqlalchemy](https://www.sqlalchemy.org)
* [nltk](http://www.nltk.org)

### Data

The provided dataset are from Figure Eight and consits of:
* **Disaster_messages.csv**     :disaster response messages from different languages  
* **Disaster_categories.csv**   :disaster messages categories

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Licensing and Acknowledgments
Thanks to Udacity and Figure Eight for the dataset
