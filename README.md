# FakeNews
Fake News Classifier

We fine-tune a BERT model to classify online news articles as Fake or True. 

The training data can be found at: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset

This project includes:
  1. A Python Notebook which can be used to train the model on Google Colab using a TPU after downloading the data from the above source and storing it on Google Drive.
  2. A GUI file that creates a GUI for users to input url's of news articles into to determine the probability of that particular article being fake or true, according to the model. Note that training must be done before the gui.py is run and the resulting PyTorch model must be saved in the same directory as gui.py.
