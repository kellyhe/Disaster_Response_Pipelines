# import libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.multioutput import MultiOutputClassifier
import pickle
from sklearn.ensemble import RandomForestClassifier
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    """ load the clean dataset from an sqlite database."""
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql("SELECT * FROM disaster_response", engine)
    X = df.message.values
    Y = df.iloc[:,4:]
    category_names = list(Y.columns)
    
    return X, Y, category_names


def tokenize(text):
    """process text data:
       tokenize and lemmatize the message 
       and remove stopwords """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        if tok not in stopwords.words("english"):
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """ Build a machine learning pipeline,
    and use grid search to find better parameters."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
         "tfidf__use_idf":[True, False] #whether to scale columns or just use normalized bag of words.
        ,"clf__estimator__n_estimators": [8, 24] # Number of trees in random forest
    }
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Show overall avg accuracy,
       and the f1 score, precision and recall for each 
       output category of the dataset."""
    y_pred = model.predict(X_test)
    accuracy = (y_pred == Y_test).mean().mean()
    print("Overall Accuracy:", accuracy)
    print("\nBest Parameters:", model.best_params_)
    
    for i in range(len(category_names)):
        print(category_names[i])
        print("\nClassification Report:\n", classification_report(Y_test.iloc[:,i], \
                  pd.DataFrame(y_pred).iloc[:,i], target_names=category_names[i]))


def save_model(model, model_filepath):
    """Export model as a pickle file"""
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
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
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()