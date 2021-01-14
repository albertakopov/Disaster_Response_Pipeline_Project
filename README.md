### Disaster Response Pipeline Project
In this project, all the data engineering skills I have learned are applied to analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages. The data set containing real messages that were sent during disaster events. A machine learning pipeline is created to categorize these events so that you can send the messages to an appropriate disaster relief agency. The project will include a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data. 

The project is made off three sections:

* **Data Processing**: an ETL (Extract, Transform and Load) pipeline is created to process all the messages and categories from the provided CSV files. The dataframe is cleaned and prepared for the ML pipeline. Finally, it is load into a SQLite database so it can be used in the next ML step

* **Machine Learning Pipeline**: splits the data into training and testing data. Feeding the data through a ML pipeline using nltk, GridSearchCV to create a supervised model. Finally, the predicted messages are classified for the 36 categaries in a MultiOutputClassifier

* **Web Application**: will display visualizations of the classified data

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
