# Thomas-Capital-Technologies
Thomas Capital Technologies
## ParseNewspaper
A module that parses newspapers (.pdf files) and extracts news by company reference. When used with the SentimentAnalysisLSTM module, it can also output sentiment analysis from an LSTM Model. See ParseNewspaperNotebook.ipynb for an example implementation.
## SentimentAnalysisLSTM
A module that trains a Long Short-Term Memory Model for Sentiment Analysis on the IMDB dataset using PyTorch. The module performs the following: 
* Accesses the IMDB dataset 
* Builds an LSTM (Long Short-Term Memory) Model
* Trains the model for text sentiment analysis using the dataset
* Classifies input text as either positive or negative
