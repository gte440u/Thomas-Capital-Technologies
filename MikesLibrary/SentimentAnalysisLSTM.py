#===========================================================
# SentimentAnalysisLSTM.py
#===========================================================
# 
# Use this module to train a LSTM Sentiment Analysis Model 
# on the IMDB dataset with PyTorch.  The module performs the
# following: 
#   -Accesses the IMDB dataset 
#   -Builds an LSTM (Long Short-Term Memory) Model
#   -Trains the model for text sentiment analysis using the 
#    dataset
#   -Classifies input text as either positive or negative
#
# Inputs:   -User-defined model parameters
#           -IMDBDataset, .csv file with review and 
#            sentiment columns
#
# Outputs:  -Trained model that can classify sentiment as 
#            positive or negative
#
# To Run:   To train a new model:
#             >>import MikesLibrary.SentimentAnalysisLSTM as SA
#             >>myTrainDL, myValidDL, myVocab = SA.processData()
#             >>myModel = SA.buildModel(myVocab)
#             >>epochs = <num_epochs_to_train_for>
#             >>myTrainedModel = SA.trainModel(myModel, myTrainDL, myValidDL, epochs)
#           To load a saved model:
#             >>import MikesLibrary.SentimentAnalysisLSTM as SA
#             >>stateDictPath = <path_to_state_dict>
#             >>vocabDictPath = <path_to_vocab_dict>
#             >>myLoadedModel = SA.loadModel(stateDictPath, vocabDictPath)
#           To perform inference:
#             >>inText = "Some positive or negative text string"
#             >>myLoadedModel.predict_sentiment(inText)
#
# Author:   Mike Thomas, June 2022
#
#===========================================================

#--------------------------------
# Python Imports
#--------------------------------
from collections import Counter                                     #Access Python Counter class from Collections librar; Dictionary-style subclass of collections with key->word and value->count of word
import string                                                       #Access Python string library
import re                                                           #Access Python regular expressions library
from datetime import datetime                                       #Access Python datetime.datetime library; for date and time info and date types
import os                                                           #Acces Python operating system library
import pickle                                                       #Access Python Pickle library; module for saving objects
import warnings                                                     #Access Python library for user warnings
                                                                    
#--------------------------------
# 3rd Party Imports aka Dependencies
#--------------------------------
from MikesLibrary.HelperFunctions import stringNow                  #Access user-defined helper function to return a string with current datetime; for archiving
import numpy as np                                                  #Access Numpy library; for linear algebra
import pandas as pd                                                 #Access Pandas library; for .csv file input/output and for data processing
import torch                                                        #Access PyTorch library
import torch.nn as nn                                               #Access PyTorch Neural Network module
from torch.utils.data import TensorDataset                          #Access PyTorch dataset wrapping tensors - each sample will be retrieved by indexing tensors
from torch.utils.data import DataLoader                             #Access PyTorch dataloader that supports map-style / iterable-style datasets and single / multi process loading
from sklearn.model_selection import train_test_split                #Access Scikit-learn ML library
from nltk.corpus import stopwords                                   #Access stopwords from NLTK library, collection for the most commonly used words (such as 'the' 'a' 'an' etc that a search engine can be programmed to ignore
                                                                    #  Works if you've downloaded stopwords from NLTK:
                                                                    #  >>import nltk                    
                                                                    #  >>nltk.download('stopwords')

#--------------------------------
# User-Defined Parameters
#--------------------------------
device = torch.device("cpu")                                                                        #set device to cpu (GPU not available)
csvPath = r"C:\Users\micha\miniconda3\envs\NeuralNetwork\data\IMDBDataset\IMDB Dataset.csv"         #set path to the IMDB database .csv file (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?resource=download)
saveFolder = r"MikesLibrary"                                                                        #folder to save trained models in
batch_size = 50                                                                                     #set batch size for loading data 
no_layers = 2                                                                                       #Number of stacked LSTMs in the NN
embedding_dim = 64                                                                                  #Number of semantic attributes to be learned per word in the embedding layer during training
hidden_dim = 256                                                                                    #Number of features in the LSTM hidden state (aka output)
output_dim = 1                                                                                      #Size of each output sample in the linear layer
drop_prob=0.5                                                                                       #probability of an input element being zeroed
lr = 0.001                                                                                          #training learning rate

#--------------------------------
# Global Objects
#--------------------------------
lTrainingLog = []                                                   #instantiate a training log; global list of strings to hold log items

#------------------------------------------
# preprocess_dataset Function Definition
#------------------------------------------
#
# Function to preprocess IMDB database into  
# split dataset before training model
# 
# Inputs:   csvPath, path to the IMDB database .csv file
#
# Outputs:  Tuple of numpy arrays consisting of the split IMDB dataset arranged as:
#             -x_train, review training dataset, numpy array
#             -x_test, review validation dataset, numpy array
#             -y_train, sentiment training dataset, numpy array
#             -y_test, sentiment validation dataset, numpy array
#
#-----------------------------------
def preprocess_dataset(csvPath):

    #--------------------------------------------------------
    # Ingest Dataset
    #--------------------------------------------------------
    #
    # Read the csv file into a pandas data frame object
    #
    #---------------------------------------------------------
    df_csvContent = pd.read_csv(csvPath)                                                

    #--------------------------------------------------------
    # Split Dataset into Training / Testing Data
    #--------------------------------------------------------
    #
    # Initially splitting the data into train and test datasets
    # avoids data leakage later
    #
    #---------------------------------------------------------
    x = df_csvContent['review'].values                                                              #pull the review column values
    y = df_csvContent['sentiment'].values                                                           #pull the sentiment column values
    x_train, x_test, y_train, y_test = train_test_split(x,y,stratify=y)                             #split data into training and testing datasets

    #--------------------------------------------------------
    # Output Training / Testing Data to User
    #--------------------------------------------------------
    #
    # Output a data preview to user:
    #   -Example movie review from IMDB database
    #   -sizes of the split datasets
    #
    #---------------------------------------------------------
    logMessage = "IMDB Reviews Sentiment Dataset Preview:\n" + df_csvContent.head().__str__()        #capture first few lines of dataframe values; default is 5 lines
    print(logMessage)                                                                                               #output message to screen
    lTrainingLog.append(" "*16 + logMessage+"\n")                                                                            #capture message in log; append to string list
    logMessage = "Example Review: " + df_csvContent.iloc[0, 0]                                                      #capture the first review in the dataset
    print(logMessage)                                                                                               #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                                                 #capture message in log; append to string list
    logMessage = "Size of review training data is " + str(len(x_train))                                             #capture size of training dataset; reviews
    print(logMessage)                                                                                               #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                                                 #capture message in log; append to string list
    logMessage = "Size of sentiment training data is " + str(len(y_train))                                          #capture size of training dataset; sentiments
    print(logMessage)                                                                                               #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                                                 #capture message in log; append to string list
    logMessage = "Size of review validation data is " + str(len(x_test))                                            #capture size of testing dataset; reviews
    print(logMessage)                                                                                               #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                                                 #capture message in log; append to string list
    logMessage = "Size of sentiment validation data is " + str(len(y_test))                                         #capture size of testing dataset; sentiments
    print(logMessage)                                                                                               #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                                                 #capture message in log; append to string list
    
    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # Return a tuple consisting of the split dataset:
    #   x_train, review training dataset
    #   x_test, review validation dataset
    #   y_train, sentiment training dataset
    #   y_test, sentiment validation dataset
    #
    #---------------------------------------------------------
    return x_train, x_test, y_train, y_test

#------------------------------------------
# preprocess_string Function Definition
#------------------------------------------
#
# Function to process strings before analyzing
# 
# Inputs:  String
#
# Outputs:  String with:
#             -non-word characters removed
#             -white space removed
#             -digits removed    
#
#-----------------------------------
def preprocess_string(s):
    
    s = re.sub(r"[^\w\s]", '', s)                                   #remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"\s+", '', s)                                       #replace whitespace with no space
    s = re.sub(r"\d", '', s)                                        #replace digits with no space
    return s                                                        #return processed string
    
#------------------------------------------
# tokenize Function Definition
#------------------------------------------
#
# Function to tokenize review datasets and encode sentiment datasetssssss
# 
# Inputs:   -x_train, review training dataset, numpy array
#           -x_val, review validation dataset, numpy array
#
# Outputs:  -tokenization of the review training dataset, numpy array of tokens which are numbers between 1 and 1000 that corrospond to the enumerated sorted word list
#           -tokenization of the review validation datasets, numpy array of tokens which are numbers between 1 and 1000 that corrospond to the enumerated sorted word list
#           -Enumerated sorted word dict of the top 1000 words from the training review dataset, dictionary with:
#              +key -> word
#              +value -> number between 1 and 1000, where 1 is the word with the highest number of occurences in the training dataset
#
#-----------------------------------
def tokenize(x_train, x_val):
    
    #--------------------------------------------------------
    # Instantiate variables
    #--------------------------------------------------------
    #
    # For temp storage of various word lists
    #
    #---------------------------------------------------------
    word_list = []                                                  #to store common words
    final_list_train = []                                           #to store final word tokens from training data
    final_list_test = []                                            #to store final word tokens from validation data
    
    #--------------------------------------------------------
    # Set stopwords
    #--------------------------------------------------------
    #
    # Set up a list of common english words (a, an, the, etc) 
    # to enable ignoring these words on saved word lists
    #
    #---------------------------------------------------------
    stop_words = set(stopwords.words('english'))
    
    #--------------------------------------------------------
    # Build Word List
    #--------------------------------------------------------
    #
    # Create a list of non-common words from the reviews 
    # training dataset
    #
    #---------------------------------------------------------
    for sent in x_train:                                            #loop thru the elements in the dataset
        for word in sent.lower().split():                           #lowercase the dataset element; loop through its words
            word = preprocess_string(word)                          #preprocess this word - clean up special chars, white space, digits
            if word not in stop_words and word != '':               #if this word is not a common word and not blank...
                word_list.append(word)                              #add this word to word list
    
    #--------------------------------------------------------
    # Create collections.Counter object of word list
    #--------------------------------------------------------
    #
    # Instantiate a collections.Counter (dictionary style) 
    # object and load it with the word list from the reviews
    # training dataset.  Words are added in the form:
    #   -key -> word
    #   -value -> count, number of instances of word 
    #    occurence in the reviews training dataset
    #
    # Counter is a dictionary-like subclass of collections 
    # used for counting hashable objects.  Elements are stored 
    # as dictionary keys and their counts are stored as 
    # dictionary values
    #
    #---------------------------------------------------------
    corpus = Counter(word_list)
    
    #--------------------------------------------------------
    # Create sorted list based on most common words
    #--------------------------------------------------------
    #
    # Use python sorted function to sort the corpus 
    # collection:
    #   -Use key to specify function to be called on the list
    #    element prior to comparison
    #   -Use .get, built-in method of dictionary object which
    #    returns the value for the key specified
    #   -Set reverse=True to sort in descending order
    #   -Return only the top 1000 words
    #   -Sorted returns a list object
    #
    #---------------------------------------------------------
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]

    #--------------------------------------------------------
    # Create dictionary based on the sorted word list
    #--------------------------------------------------------
    #
    # Create a dictionary object:
    #   -Use enumerate to iterate through the sorted list, 
    #    returning a counter and list value at the same time
    #   -enumerate count starts at 0, add 1 to start 
    #    dictionary values at 1
    #   -set key -> word
    #   -set value -> the enumerate count index, ranges 
    #    from 1 to 1000 with 1 being the most commonly 
    #    occuring word (based on the first entry in the 
    #    descending count sorted word list) and 1000 being 
    #    the 1000th most common 
    #   -Returns a dict object
    #
    #---------------------------------------------------------
    word_list_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    #--------------------------------------------------------
    # Create tokenized review training dataset
    #--------------------------------------------------------
    #
    # Loop thru the reviews in the training dataset, and thru 
    # the words in each review, to create a tokenized list
    # for each review:
    #   -For each word return its token if the word is found
    #    in the sorted word list dictionary
    #   -Tokens are numbers between 1 and 1000, with 1 
    #    corrosponding to the most commonly occuring word
    #   -Returns a list object where each element corrosponds 
    #    to a review
    #   -Each review is a list object where each element 
    #    corrosponds to the token corrosponding to the word
    #    in the review
    #
    #---------------------------------------------------------
    for sent in x_train:
        final_list_train.append([word_list_dict[preprocess_string(word)] for word in sent.lower().split() if preprocess_string(word) in word_list_dict.keys()])
        
    #--------------------------------------------------------
    # Create tokenized review validation dataset
    #--------------------------------------------------------
    #
    # Loop thru the reviews in the validation dataset, and 
    # thru the words in each review, to create a tokenized 
    # list for each review:
    #   -For each word return its token if the word is found
    #    in the sorted word list dictionary
    #   -Tokens are numbers between 1 and 1000, with 1 
    #    corrosponding to the most commonly occuring word
    #   -Returns a list object where each element corrosponds 
    #    to a review
    #   -Each review is a list object where each element 
    #   -corrosponds to the token corrosponding to the word
    #    in the review
    #
    #---------------------------------------------------------
    for sent in x_val:
        final_list_test.append([word_list_dict[preprocess_string(word)] for word in sent.lower().split() if preprocess_string(word) in word_list_dict.keys()])

    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # Return a tuple consisting of:
    #   tokenized review training dataset; numpy array
    #   tokenized review validation dataset; numpy array
    #   Enumerated sorted word dict
    #
    #---------------------------------------------------------
    return np.array(final_list_train), np.array(final_list_test), word_list_dict
    
#------------------------------------------
# encode Function Definition
#------------------------------------------
#
# Function to encode sentiment datasets
# 
# Inputs:   y_train, sentiment training dataset, numpy array
#           y_test, sentiment validation dataset, numpy array
#
# Outputs:  encoded sentiment training dataset (positive=1, negative=0), numpy array
#           encoded validation sentiment dataset (positive=1, negative=0), numpy array
#           
#-----------------------------------
def encode(y_train, y_val):

    #--------------------------------------------------------
    # Encode sentiment
    #--------------------------------------------------------
    #
    # Encode sentiment in training and validation datasets as:
    #   Convert positive sentiment to 1
    #   Convert negative sentiment to 0
    #
    #---------------------------------------------------------
    encoded_train = [1 if label == 'positive' else 0 for label in y_train]
    encoded_test = [1 if label == 'positive' else 0 for label in y_val]
    
    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # Return a tuple consisting of:
    #   encoded sentiment training dataset
    #   encoded sentiment validation dataset
    #
    #---------------------------------------------------------
    return np.array(encoded_train), np.array(encoded_test)

#---------------------------------------------------------
# padding_ Function Definition
#---------------------------------------------------------
#
# Function to pad/cut sequences to max length
# 
# Inputs:   -sentences, tokenized dataset, numpy array 
#           -seq_len, length to pad or cut the sentences to, 
#            int
#
# Outputs:  -tokenized dataset padded up or cut down to 
#            the seq_len
#
#---------------------------------------------------------
def padding_(sentences, seq_len):
    
    #--------------------------------------------------------
    # Create Zeros Array
    #--------------------------------------------------------
    #
    # Create a numpy array:
    #   -#rows equals #reviews
    #   -#columns set to 500
    #   -populated with all 0s
    #
    #---------------------------------------------------------
    features = np.zeros((len(sentences), seq_len), dtype=int)
    
    #--------------------------------------------------------
    # Tokenized reviews loop
    #--------------------------------------------------------
    #
    # Loop thru the tokenized reviews:
    #   -#rows equals #reviews
    #   -#columns set to 500
    #   -populated with all 0s
    #
    #---------------------------------------------------------
    for ii, review in enumerate(sentences):
    
        #if review is not blank...
        if len(review) != 0:
        
            #--------------------------------------------------------
            # Create padded array
            #--------------------------------------------------------
            #
            # Add this tokenized review to the features array:
            #   -Max size of review is capped at seq_len
            #   -Place at the end of the array, by wrapping 
            #    from -len(review) to the 0 index and thus padding 
            #    with zeros from the start of the array
            #   -Note that with numpy arrays, if you wrap from 
            #    negative further back than the array's length, it 
            #    simply starts from 0
            #
            #---------------------------------------------------------
            features[ii, -len(review):] = np.array(review)[:seq_len]
            
    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # Padded tokenized dataset
    #
    #---------------------------------------------------------
    return features

#---------------------------------------------------------
# SentimentRNN Class Definition
#---------------------------------------------------------
#
# Class to establish a RNN to perform sentiment analysis
# 
# Inherits from:  nn.Module
# 
# Methods:
#   -__init__ -> initialize class instantiation
#   -forward -> define recipe for forward pass
#   -init_hidden -> initialize LSTM hidden state
#
#---------------------------------------------------------
class SentimentRNN(nn.Module):

    #--------------------------------------------------------
    # __init__ Method Definition
    #--------------------------------------------------------
    #
    # Class instatiation initialization method
    # 
    # Inputs:   -no_layers, number of stacked LSTMs in the 
    #            NN, int
    #           -vocab, enumerated sorted word dict, dict   
    #           -hidden_dim, number of features in the LSTM 
    #            hidden state aka output, int
    #           -embedding_dim, number of semantic attributes 
    #            to be learned per word in the embedding 
    #            layer during training, int
    #           -output_dim, size of each output sample in 
    #            the linear layer, int
    #           -drop_prob, probability of an input element
    #            being zeroed, float
    #
    # Outputs:  -initialized instantiation of 
    #            SentimentRNN class
    #           
    #--------------------------------------------------------
    def __init__(self, no_layers, vocab, hidden_dim, embedding_dim, output_dim, drop_prob, lr):
    
        #return an object that represents the parent class to avoide having to use base class name explicitly
        super(SentimentRNN, self).__init__()
        
        #--------------------------------------------------------
        # Initialize model parameters
        #--------------------------------------------------------
        #
        # Save model parameters to the instantiated SentimentRNN
        # as attributes 
        #
        #---------------------------------------------------------
        self.no_layers = no_layers
        self.vocab_size = len(vocab) + 1
        self.vocab = vocab
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.drop_prob = drop_prob
        self.lr = lr
        
        #--------------------------------------------------------
        # Set Additional Model Attributes
        #--------------------------------------------------------
        #
        # Additional helpful model attributes
        #
        #---------------------------------------------------------
        self.save_dir = saveFolder
        
        #--------------------------------------------------------
        # Initialize embedding layer
        #--------------------------------------------------------
        #
        # Initialize model embedding layer:
        #   -of class torch.nn.Embedding, a simple lookup table 
        #    that stores embeddings of a fixed dictionary and 
        #    size
        #   -num_embeddings, size of the dictionary of 
        #    embeddings aka the number of words you'd like the NN
        #    to learn semantic attributes about, is set to the 
        #    length of the enumerated sorted word dict, int
        #   -embedding_dim, size of each embedding vector aka
        #    number of semantic attributes to be learned per 
        #    word, int
        #     
        # This architecture utilizes word embeddings, where the 
        # embeddings are a representation of the semantics of a 
        # word that might be relevant to the task at hand.  The 
        # embeddings take the form of a vector for each word, 
        # with each vector made up of semantic attributes.  An 
        # example of a semantic attribute might be things that 
        # can run; humans would score high on this attribute 
        # and rocks would score low.  Each attribute is a 
        # dimension, and so the number of attributes you allow
        # the NN to learn per word is the size of the embedding
        # vector aka the embedding dimension.  The NN learns 
        # these representations during training because the 
        # embeddings are parameters in the model.  
        #
        #---------------------------------------------------------
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        
        #--------------------------------------------------------
        # Initialize LSTM layer
        #--------------------------------------------------------
        #
        # Initialize model long-short term memory layer:
        #   -of class torch.nn.LSTM, which applies a multi-layer 
        #    LSTM RNN to an input sequence
        #   -input_size, number of expected features in the input, 
        #    set to embedding_dim aka number of semantic  
        #    attributes to be learned per word in the embedding
        #    layer, int 
        #   -hidden_size, number of features in the hidden state 
        #    aka the output, set to hidden_dim, int
        #   -num_layers, number of recurrent LSTM layers, is set
        #    to no_layers, int
        #   -batch_first, set to True such that the input and 
        #    output tensors are provided as (batch, seq, feature)
        #    as opposed to (seq, batch, feature)
        #
        # Setting num_layers = 2 means stacking two LSTMs together 
        # to form a stacked LSTM, with the second LSTM taking in 
        # outputs of the first LSTM and computing the final 
        # results
        #
        #---------------------------------------------------------
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_dim, num_layers=self.no_layers, batch_first=True)
        
        #--------------------------------------------------------
        # Initialize dropout layer
        #--------------------------------------------------------
        #
        # Initialize model dropout layer:
        #   -of class torch.nn.Dropout, during training randomly 
        #    zeroes some elements of the input tensor with 
        #    probability p
        #   -p, probability of an element to be zeroed; float
        #
        # Randomly zeroing out input elements has been found to 
        # reduce model overfitting
        #
        #---------------------------------------------------------
        self.dropout = nn.Dropout(self.drop_prob)
        
        #--------------------------------------------------------
        # Initialize linear layer
        #--------------------------------------------------------
        #
        # Initialize model linear layer:
        #   -of class torch.nn.Linear, applies a linear 
        #    transformation to the incoming data
        #   -in_features, size of each input sample, set to 
        #    hidden_dim aka number of features in the LSTM hidden 
        #    state aka output, int
        #   -out_features, size of each output sample, set to 
        #    output_dim, int
        #
        #---------------------------------------------------------
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        
        #--------------------------------------------------------
        # Initialize sigmoid layer
        #--------------------------------------------------------
        #
        # Initialize model sigmoid activation function layer:
        #   -of class torch.nn.Sigmoid, applies element-wise 
        #    sigmoid function
        #
        # The Sigmoid function produces the characteristic 
        # S-shaped curve
        #
        #---------------------------------------------------------
        self.sig = nn.Sigmoid()
        
    #--------------------------------------------------------
    # forward Method Definition
    #--------------------------------------------------------
    #
    # Define recipe for the forward pass; forward method 
    # accepts a tensor of input data and returns a tensor of 
    # output data
    # 
    # Inputs:   -x, inputs 
    #           -hidden state 
    #
    # Outputs:  -sig_out, last sigmoid output
    #           -hidden, LSTM hidden state aka output
    #
    # While this method defines the computation performed at 
    # every call; one should call the module instance instead
    # of this since the module instance takes care of running
    # the registered hooks but the forward pass method 
    # ignores them
    # 
    # nn.Module.forward should be overridden by all 
    # subclasses (aka child classes inhereiting from 
    # nn.Module)
    # 
    # The forward method can access methods defined in the 
    # initialization method.
    #           
    #--------------------------------------------------------
    def forward(self, x, hidden):
        
        #--------------------------------------------------------
        # Pull batch size
        #--------------------------------------------------------
        #
        # Pull size along the zero dimension of the input tensor
        #
        #---------------------------------------------------------
        batch_size = x.size(0)
        
        #--------------------------------------------------------
        # Embeddings Layer
        #--------------------------------------------------------
        #
        # Simple lookup table that stores embeddings of a fixed 
        # dictionary and size
        #   -aka semantic vector attributes
        #   -shape: B x S X Features since batch = True
        #
        #---------------------------------------------------------
        embeds = self.embedding(x)
        
        #--------------------------------------------------------
        # LSTM Layer
        #--------------------------------------------------------
        #
        # In an LSTM, for each element in the sequence there is a
        # corresponding hidden state which can contain 
        # information from arbitrary points earlier in the 
        # sequence; these hidden states can be used for 
        # predictions
        #
        # Pull LSTM output using model stacked lstm layers:
        #   Pass in:
        #     -inputs => embeddings
        #     -hidden => initialized hidden state
        #   Return:
        #     -lstm_out => all of the hidden states throughout 
        #                  the sequence
        #     -hidden => most recent hidden state, aka last slice
        #                of lstm_out
        #
        #---------------------------------------------------------
        lstm_out, hidden = self.lstm(embeds, hidden)
        
        #--------------------------------------------------------
        # Process LSTM Layer Output
        #--------------------------------------------------------
        #
        # Use tensor.contiguous() method to set desired format of
        # returned tensor to contiguous, a contiguous in memory 
        # tensor
        # 
        # Use tensor.view() method; ouptut tensor shares the same 
        # underlying data with its base tensor, allows fast and
        # memory efficient reshaping slicing and element-wise 
        # oeprations
        #
        #---------------------------------------------------------
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
        
        #--------------------------------------------------------
        # Dropout Layer
        #--------------------------------------------------------
        #
        # Feed processed LSTM output through dropout layer; 
        # randomly zeroes some of the elements of the input 
        # tensor
        #
        #---------------------------------------------------------
        out = self.dropout(lstm_out)
        
        #--------------------------------------------------------
        # Linear Layer
        #--------------------------------------------------------
        #
        # Feed dropout layer output through fully connected 
        # linear layer; applies linear transformation to the 
        # input tensor
        #
        #---------------------------------------------------------
        out = self.fc(out)
        
        #--------------------------------------------------------
        # Sigmoid Layer
        #--------------------------------------------------------
        #
        # Feed linear layer output through sigmoid layer; applies
        # sigmoid function to the input tensor
        #
        #---------------------------------------------------------
        sig_out = self.sig(out)
        
        #--------------------------------------------------------
        # Process Sigmoid Layer Output
        #--------------------------------------------------------
        #
        # -Reshape sigmoid output to be batch_size first
        # -Pull last batch of labels from sigmoid output
        #
        #---------------------------------------------------------
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        
        #--------------------------------------------------------
        # Method Returns
        #--------------------------------------------------------
        #
        # tuple of:
        #   -last batch of labels from sigmoid output
        #   -most recent hidden state from LSTM layer
        #
        #---------------------------------------------------------
        return sig_out, hidden
        
    #--------------------------------------------------------
    # init_hidden Method Definition
    #--------------------------------------------------------
    #
    # Method to initialize the LSTM hidden state
    # 
    # Inputs:   -batch_size, int 
    #
    # Outputs:  -hidden, tuple of:
    #              +h0, hidden state tensor initialized to 0
    #              +c0, cell state tensor initialized to 0
    #           
    #--------------------------------------------------------
    def init_hidden(self, batch_size):
    
        #--------------------------------------------------------
        # Initialize hidden state tensors
        #--------------------------------------------------------
        #
        # Initialize hidden state of LSTM from two identical 
        # tensors h0 and c0:
        #   -of class torch.Tensor
        #   -h0 for hidden state
        #   -c0 for cell state
        #   -size:  no_layers x batch_size x hidden_dim
        #     +no_layers, number of stacked LSTMs in the NN, int
        #     +batch_size, size of batch, int
        #     +hidden_dim, number of features in the LSTM hidden 
        #      state aka output, int
        #   -initialize tensors to 0
        #   -return tensors as tuple
        #
        #---------------------------------------------------------
        h0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers, batch_size, self.hidden_dim)).to(device)
        hidden = (h0, c0)
        
        #--------------------------------------------------------
        # Method Returns
        #--------------------------------------------------------
        #
        # hidden, tuple of:
        #   -h0, hidden state tensor initialized to 0
        #   -c0, cell state tensor initialized to 0
        #
        #---------------------------------------------------------
        return hidden
        
    #--------------------------------------------------------
    # save Method Definition
    #--------------------------------------------------------
    #
    # Method to save model:
    #
    #   Save the model state dictionary.  This serializes 
    #   the internal state dictionary, containing the model 
    #   parameters, to save the model.
    #
    #   Save the model vocabulary as a pickle dump
    # 
    # Utilize .pt extension rather than .pth extension to 
    # avoid collisions with python path .pth configuration 
    # files. Note that .pt and .pth are the same format based
    # on the litarature.
    # 
    # Inputs:   -self 
    #
    # Outputs:  -paths to saved model dict and vocab
    #           
    #--------------------------------------------------------
    def save(self):
    
        #--------------------------------------------------------
        # Build File Save Paths
        #--------------------------------------------------------
        #
        # -Model State Dictionary
        # -Model Vocabulary
        #
        #---------------------------------------------------------
        sNow = stringNow()                                                                                  #Pull now in string format; for datetime unique save paths
        sRunningPyFileName = os.path.basename(__file__)[:-3]                                                #Pull the name of this python file; remove .py file extension
        sDictSavePath = self.save_dir + '\\' + sNow + "_" + sRunningPyFileName + "_modelStateDict.pt"       #Build path to save model state dictionary      
        sVocabSavePath = self.save_dir + '\\' + sNow + "_" + sRunningPyFileName + "_vocab.pkl"              #Build path to save model vocabulary
        
        #--------------------------------------------------------
        # Save Model
        #--------------------------------------------------------
        #
        # -Model State Dictionary
        # -Model Vocabulary
        #
        #---------------------------------------------------------
        torch.save(self.state_dict(), sDictSavePath)                #save model state dict
        with open(sVocabSavePath, "wb") as f:                       #open model vocabulary path for write; creates file if it doesnt exist, overwrites if it does exist
            pickle.dump(self.vocab, f)                              #perform pickel dump to save model vocab dict object

        #--------------------------------------------------------
        # Output Save Info to User
        #--------------------------------------------------------
        #
        # Output to user:
        #   -Model State Dictionary saved
        #   -Model Vocabulary saved
        #
        #---------------------------------------------------------
        logMessage = stringNow() + " Saving Model"                                      #saving model user message
        print(logMessage)                                                               #output message to screen
        lTrainingLog.append(logMessage + "\n")                                          #capture message in log; append to string list
        logMessage = "Saved PyTorch Model State Dictionary to " + sDictSavePath         #saving model state dictionary user message
        print(logMessage)                                                               #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")                                 #capture message in log; append to string list
        logMessage = "Saved PyTorch Model Vocab Dictionary to " + sVocabSavePath        #saving model vocabulary user message
        print(logMessage)                                                               #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")                                 #capture message in log; append to string list
        
    #--------------------------------------------------------
    # predict_sentiment Method Definition
    #--------------------------------------------------------
    #
    # Method to perform inference from model and predict 
    # sentiment of input text
    # 
    # Inputs:   -self
    #           -text, input to perform inference on; str
    #
    # Outputs:  -Method Returns a tuple of:
    #             -inference probability, where > 0.5 is 
    #              positive sentiment and < 0.5 is negative 
    #              sentiment; float
    #             -predicted sentiment (positive or 
    #              negative); string
    #             -sentiment confidence, confidence in 
    #              predicted sentiment being correct; float
    #           -Prints sentiment assessment to terminal
    #
    #--------------------------------------------------------
    def predict_sentiment(self, text):
        
        #--------------------------------------------------------
        # Parse Input Text into Array
        #--------------------------------------------------------
        #
        # Parse the input text; returning an array with each 
        # word's ranking from the vocabulary. The vocabulary is 
        # an enumerated sorted word dict of the top 1000 words 
        # from the training review dataset; each word's value is 
        # a number where 1 is the word with the highest number of 
        # occurences in the training dataset
        #
        # -Use str.split() method to seperate the input text 
        #  string into a word list of strings
        # -Loop through the words in the word list
        # -Pass this word through preprocess_string to clean up 
        #  special chars, white space, digits
        # -If this preprocessed word is in the vocabulary:
        #    -Pass the word through preprocess_string to clean up
        #     special chars, white space, digits
        #    -Lookup this preprocessed word (aka key) in vocab dict
        #     and return its associated occurance rank (aka 
        #     value); returned as int
        # -Wrap the resulting returned found word ranks in a list;
        #  list of ints
        # -Wrap the ranks list in a numpy array
        # -Shape of array: <num_words> rows x 0 columns 
        #
        #---------------------------------------------------------
        word_seq = np.array([self.vocab[preprocess_string(word)] for word in text.split() if preprocess_string(word) in self.vocab.keys()])
        
        #--------------------------------------------------------
        # Expand Array
        #--------------------------------------------------------
        #
        # Expand numpy array
        #
        # -Use numpy .expand_dims() method to expand array by 
        #  inserting new axis at 0 axis position
        # -Shape of array: 1 rows x <num_words> columns 
        #
        #---------------------------------------------------------
        word_seq = np.expand_dims(word_seq, axis=0)
        
        #--------------------------------------------------------
        # Perform data padding on Array
        #--------------------------------------------------------
        #
        # Pad array with zeros to get word sequence arrays of 
        # same lengths
        #
        #---------------------------------------------------------
        pad = torch.from_numpy(padding_(word_seq, 500))
        
        #--------------------------------------------------------
        # Move to device
        #--------------------------------------------------------
        #
        # Move padded word sequence array to device
        #
        #---------------------------------------------------------
        inputs = pad.to(device)
        
        #--------------------------------------------------------
        # Initialize Hidden State
        #--------------------------------------------------------
        #
        # Initialize the model's LSTM layer hidden state; returns
        # a tuple of zerod tensors
        #
        #---------------------------------------------------------
        batch_size = 1                                              #Use batch_size of 1 for inference
        h = self.init_hidden(batch_size)                            #initialize hidden state
        
        #--------------------------------------------------------
        # Unpack the hidden states tuple
        #--------------------------------------------------------
        #
        # For each tensor in the hidden state tuple:
        #   -Call PyTorch tensor.data
        #   -This allows for in-place updates to the tensors
        #   -This sets requires_grad=False; changes to tensors 
        #    are not being tracked by autograd
        # Wrap the tensors in a list
        # Wrap the list of tensors in a tuple
        #
        # Note: This may be an obsolete coding holdover from when 
        # PyTorch Variable objects and PyTorch Tensor objects 
        # were seperate; the Variable .data attribute was used to 
        # access the underlying Tensor structure from the Variable
        # object. Per current PyTorch documentation its use is not
        # recommended; keeping in for completeness 
        #---------------------------------------------------------
        h = tuple([each.data for each in h])
        
        #--------------------------------------------------------
        # Call the Model - Perform Forward Pass
        #--------------------------------------------------------
        #
        # This step calls the forward method from the 
        # SentimentRNN class description which takes in a 
        # text batch and the current LSTM hidden state, performs 
        # a forward pass through the NN architecture, and outputs 
        # the results from the sigmoid layer and the LSTM hidden 
        # state
        #
        # This step can take up to 1min per forward pass
        #
        #---------------------------------------------------------
        output, h = self(inputs, h)
        
        #--------------------------------------------------------
        # Calculate Results
        #--------------------------------------------------------
        #
        # Pull inference probability from model output and 
        # calculate sentiment and sentiment confidence
        #
        #---------------------------------------------------------
        infPro = output.item()                                                                              #pull model inference probability outputfrom tensor; returned as float
        sent = "positive" if infPro > 0.5 else "negative"                                                   #if inference probability is greater than .5 positive sentiment; otherwise negative
        if sent == "positive": sentConf = (infPro - 0.5)/0.5                                                #calculate confidence if positive (how close to 1.0, within the interval 0.5 to 1.0, where 0.5 is unsure and 1.0 is definitely positive, is this infPro?)
        elif sent == "negative": sentConf = (0.5 - infPro)/0.5                                              #calculate confidence if negative (how close to 0, within the interval 0.5 to 0, where 0.5 is unsure and 0 is definitely negative, is this infPro?)
        else: warnings.warn(f"Couldnt determine sentiment; inferrence probability => {infPro} ")            #user warning...couldnt determine sentiment...this shouldnt happen...
        
        #--------------------------------------------------------
        # Method Return
        #--------------------------------------------------------
        #
        # Method returns tuple of:
        #   -infPro aka inference probability, where > 0.5 is 
        #    positive sentiment and < 0.5 is negative sentiment; 
        #    float
        #   -sent aka predicted sentiment (positive or negative);
        #    string
        #   -sentConf aka sentiment confidence, confidence in 
        #    predicted sentiment being correct; float
        #
        #---------------------------------------------------------
        return infPro, sent, sentConf

#---------------------------------------------------------
# processData Function Definition
#---------------------------------------------------------
#
# Function to process data before ingesting into the model
# for training
# 
# Inputs:   -csvPath, path to the IMDB database .csv file
#
# Outputs:  -Training dataloader, of class 
#            torch.utils.data.DataLoader
#           -Validation dataloader, of class 
#            torch.utils.data.DataLoader
#           -Enumerated sorted word dict of the top 1000 
#            words from the training review dataset, 
#            dictionary with:
#              +key -> word
#              +value -> number between 1 and 1000, where 
#               1 is the word with the highest number of 
#               occurences in the training dataset 
#
#---------------------------------------------------------
def processData():

    #--------------------------------------------------------
    # Perform data preprocessing
    #--------------------------------------------------------
    #
    # Preprocess the IMDB database by splitting into:
    #   -Training Dataset:
    #      +reviews
    #      +sentiment
    #   -Validation Dataset:
    #      +Reviews
    #      +Sentiment
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Performing data processing"            #user message
    print(logMessage)                                                   #output message to screen
    lTrainingLog.append(logMessage + "\n")                              #capture message in log; append to string list
    x_train, x_test, y_train, y_test = preprocess_dataset(csvPath)      #Split database into training / validation datasets                                        
    
    #--------------------------------------------------------
    # Perform data tokenization
    #--------------------------------------------------------
    #
    # Tokenize review datasets and create vocab dictionary
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Performing data tokenization"      #user message
    print(logMessage)                                               #output message to screen
    lTrainingLog.append(logMessage + "\n")                          #capture message in log; append to string list
    x_train, x_test, vocab = tokenize(x_train, x_test)              #Tokenize review datasets
    
    #--------------------------------------------------------
    # Perform data encoding
    #--------------------------------------------------------
    #
    # Encode sentiment datasets
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Performing data encoding"          #user message
    print(logMessage)                                               #output message to screen
    lTrainingLog.append(logMessage + "\n")                          #capture message in log; append to string list
    y_train, y_test = encode(y_train, y_test)                       #Encode sentiment datasets    
    
    #--------------------------------------------------------
    # Perform data padding
    #--------------------------------------------------------
    #
    # Pad tokenized review datasets with zeros to get 
    # tokenized reviews of same lengths
    #
    #---------------------------------------------------------
    x_train_pad = padding_(x_train, 500)                            #Pad review training dataset
    x_test_pad = padding_(x_test, 500)                              #Pad review validation dataset
    
    #--------------------------------------------------------
    # Create tensor datasets
    #--------------------------------------------------------
    #
    # Create tensor datasets by wrapping datasets, of class
    # torch.utils.data.TensorDataset
    #
    #---------------------------------------------------------
    train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))        #create training tensor dataset
    valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))          #create validation tensor dataset
    
    #--------------------------------------------------------
    # Create dataloaders
    #--------------------------------------------------------
    #
    # Create iterable dataloaders from tensor dataset:
    #   -Set to reshufle data at every epoch
    #   -Set to load batch_size samples per batch
    #   -Of class torch.utils.data.DataLoader
    #
    #---------------------------------------------------------
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)                  #create training dataloader
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)                  #create validation dataloader
    
    #--------------------------------------------------------
    # User Information
    #--------------------------------------------------------
    #
    # Output data processing complete user message
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Data processing complete"          #user message
    print(logMessage + "\n" + "-"*70)                               #output message to screen
    lTrainingLog.append(logMessage + "\n")                          #capture message in log; append to string list
    
    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # Return a tuple consisting of:
    #   -Training dataloader, of class 
    #    torch.utils.data.DataLoader
    #   -Validation dataloader, of class
    #    torch.utils.data.DataLoader
    #   -Enumerated sorted word dict of the top 1000 
    #    words from the training review dataset, 
    #    dictionary with:
    #      +key -> word
    #      +value -> number between 1 and 1000, where 
    #       1 is the word with the highest number of 
    #       occurences in the training dataset 
    #
    #---------------------------------------------------------
    return train_loader, valid_loader, vocab
    
#---------------------------------------------------------
# buildModel Function Definition
#---------------------------------------------------------
#
# Function to build the sentiment analysis model
# 
# Inputs:   -User-defined parameters
#           -vocab, enumerated sorted word dict of the top
#            1000 words from the training review dataset, 
#            dictionary with:
#              +key -> word
#              +value -> number between 1 and 1000, where 
#               1 is the word with the highest number of 
#               occurences in the training dataset
#
# Outputs:  -Untrained sentiment analysis model
#
#---------------------------------------------------------
def buildModel(vocab):

    #--------------------------------------------------------
    # Instantiate model
    #--------------------------------------------------------
    #
    # Instantiate an instance of the SentimentRNN and move 
    # it to device
    #
    #---------------------------------------------------------
    model = SentimentRNN(no_layers, vocab, hidden_dim, embedding_dim, output_dim, drop_prob, lr)
    model.to(device)

    #--------------------------------------------------------
    # User Information
    #--------------------------------------------------------
    #
    # Output information to user about the built model, will
    # be needed when later comparing loaded models
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Built model architecture:\n" + model.__str__()         #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(logMessage + "\n")                                              #capture message in log; append to string list
    logMessage = "Model parameters:"                                                    #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                     #capture message in log; append to string list
    logMessage = "LSTM layers: " + str(no_layers)                                       #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                     #capture message in log; append to string list
    logMessage = "Vocabulary size: " + str(len(vocab) + 1)                              #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                     #capture message in log; append to string list
    logMessage = "LSTM hidden dimension: " + str(hidden_dim)                            #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                     #capture message in log; append to string list
    logMessage = "Embedding dimension: " + str(embedding_dim)                           #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                     #capture message in log; append to string list
    logMessage = "Drop probability: " + str(drop_prob)                                  #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(" "*16 + logMessage + "\n")                                     #capture message in log; append to string list
    logMessage = stringNow() + " Model build complete"                                  #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(logMessage + "\n")                                              #capture message in log; append to string list
    
    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # Untrained sentiment analysis model
    #
    #---------------------------------------------------------
    return model

#---------------------------------------------------------
# acc Function Definition
#---------------------------------------------------------
#
# Function to help predict model output accuracy
# 
# Inputs:   -pred, prediction output from the model, of 
#            class torch.Tensor
#           -labels, encoded sentiment data labels, of 
#            class torch.Tensor
#
# Outputs:  Summed comparison of True (1) and False (0) 
#           comparisons between model predicted sentiment 
#           and labeled sentiment, int result of a 
#           torch.sum().item
#
#---------------------------------------------------------
def acc(pred, labels):

    #--------------------------------------------------------
    # Preprocess prediction
    #--------------------------------------------------------
    #
    # squeeze -> remove single-dimension entries from the 
    #            shape of the array
    #
    # round ->   round elements of the array to the nearest 
    #            integer, this rounds the predictions to 0 or 
    #            1 so they can be compared to the encoded 
    #            sentiment labels
    #
    #---------------------------------------------------------
    pred = torch.round(pred.squeeze())
    
    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # The logic comparison compares the sentiment predictions
    # from the model (0 for negative, 1 for positive) to the 
    # encoded sentiment data labels (also 0 for negative and 
    # 1 for positive) and returns True or False depending on
    # a match
    #
    # torch.sum returns the sum of all elements in the input 
    # tensor (in this case the logic comparison tensor, with 1
    # for True and 0 for False); and .item() returns this sum 
    # as int.  
    #
    #---------------------------------------------------------
    return torch.sum(pred == labels.squeeze()).item()
    
#---------------------------------------------------------
# trainModel Function Definition
#---------------------------------------------------------
#
# Function to train the sentiment analysis model
# 
# Inputs:   -User-defined parameters
#           -Untrained sentiment analysis model
#           -training data loader
#           -validation data loader
#           
#
# Outputs:  -Trained sentiment analysis model
#
#---------------------------------------------------------
#def trainModel(model, train_loader, valid_loader):
def trainModel(model, train_loader, valid_loader, epochs):

    #--------------------------------------------------------
    # Setup loss function
    #--------------------------------------------------------
    #
    # Setup loss function:
    #   -Create a criterion that measures the binary cross 
    #    entropy between the target and the input 
    #    probabilities
    #   -of class torch.nn.BCELoss
    #
    #---------------------------------------------------------
    criterion = nn.BCELoss()
    
    #--------------------------------------------------------
    # Setup optimization algorithm
    #--------------------------------------------------------
    #
    # Implement optimization algorithm:
    #   -Construct an optimizer object that will hold the 
    #    current state and will update the parameters based
    #    on the computed gradients
    #   -Pass the optimizer an iterable containing parameters
    #    to optimize, in this case the model parameters
    #      +Specifically, the optimizer can only optimize 
    #       iterators of type tensor   
    #   -Specify optimizer learning rate at .001 (default)
    #   -Implement Adam optimization algorithm, an algorithm 
    #    for first-order gradient-based optimization of 
    #    stochastic objective functions based on adaptive 
    #    estimates of lower-order moments
    #   -of class torch.optim.Adam
    # 
    # model.parameters() returns an iterator over module 
    # parameters, and by default includes parameters of all
    # submodules.  Specifically the iterator returned is of 
    # class generator.  Generator functions allow for 
    # creation of user-defined functions that behave like an 
    # iterator; implemented by using the keyword yield as 
    # opposed to return in func def.  
    #
    # For this model architecture, model.parameters() contains
    # Embedding layer weights, LSTM layers weights and biases,
    # and fc (aka linear transform layer) weights and biases
    #
    #---------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr)
    
    #--------------------------------------------------------
    # Set training parameters
    #--------------------------------------------------------
    #
    # Set model training parameters
    #
    #---------------------------------------------------------
    clip = 5                                                        #clip value; for clip_grad_norm
    #epochs = 5                                                      #number of epochs
    valid_loss_min = np.Inf                                         #set validation loss minimum to infinity
    epoch_tr_loss = []                                              #instantiate list for training loss
    epoch_vl_loss = []                                              #instantiate list for validation loss
    epoch_tr_acc = []                                               #instantiate list for training accuracy
    epoch_vl_acc = []                                               #instantiate list for validation accuracy

    #--------------------------------------------------------
    # User Information
    #--------------------------------------------------------
    #
    # Output training start message
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Begining training"                 #user message
    print("\n" + "-"*70 + "\n" + logMessage)                        #output message to screen
    lTrainingLog.append(logMessage + "\n")                          #capture message in log; append to string list

    #--------------------------------------------------------
    # Training loop
    #--------------------------------------------------------
    #
    # Loop over epochs to train the model
    #
    #---------------------------------------------------------
    for epoch in range (epochs):

        #--------------------------------------------------------
        # Set up model for training
        #--------------------------------------------------------
        #
        # Set up model for the training epoch loop
        #
        #---------------------------------------------------------
        model.train()                                               #Set the model to training mode
        
        #--------------------------------------------------------
        # Initialize trainnig loop objects
        #--------------------------------------------------------
        #
        # Initialize objects for this training epoch loop
        #
        #---------------------------------------------------------
        train_losses = []                                           #List to hold loss values during training
        train_acc = 0.0                                             #Variable to hold model accuracies vs truth during training
        h = model.init_hidden(batch_size)                           #Initialize the model's hidden state; returns tuple of zerod tensors
        train_loader_counter = 0                                    #instantiate train loader counter
        
        #--------------------------------------------------------
        # User Information
        #--------------------------------------------------------
        #
        # Output training run time user message
        #
        #---------------------------------------------------------
        logMessage = stringNow() + " Begining training data loader loop Epoch " + str(epoch+1)                                  #user message
        print("-"*70 + "\n" + logMessage)                                                                                              #output message to screen
        lTrainingLog.append(logMessage + "\n")                                                                                  #capture message in log; append to string list
        logMessage = "This step takes up to 2mins per element, train_loader elements to process: " + str(len(train_loader))     #user message
        print(logMessage)                                                                                                       #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")                                                                         #capture message in log; append to string list
        logMessage = "Processing train_loader elements: "                                                                       #user message
        print(logMessage)                                                                                                       #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")                                                                         #capture message in log; append to string list
        
        #--------------------------------------------------------
        # Training data loop
        #--------------------------------------------------------
        #
        # Loop over tokenized reviews & encoded sentiments 
        # batches in the training data loader
        #
        #---------------------------------------------------------
        for tokenizedReviews, encodedSentiments in train_loader:
            
            #--------------------------------------------------------
            # User Information
            #--------------------------------------------------------
            #
            # Output element counter user message
            #
            #---------------------------------------------------------
            logMessage = stringNow() + " Processing element => " + str(train_loader_counter)        #user message
            print(logMessage)                                                                       #output message to screen
            lTrainingLog.append(logMessage + "\n")                                                  #capture message in log; append to string list
        
            #--------------------------------------------------------
            # Use for debugging
            #--------------------------------------------------------
            #
            # This is error-checking code only:
            #   -break out of loop after completing 2 elements
            #   -use to check code in a reasonable amount of time
            #
            #---------------------------------------------------------
            if train_loader_counter > 2:
               print(f"Debugging:  Breaking train_loader loop at element: {train_loader_counter}")
               break
            
            #--------------------------------------------------------
            # Move to device
            #--------------------------------------------------------
            #
            # Move training data tensors to device in anticipation of 
            # calculations:
            #   -this batch of tokenized reviews
            #   -this batch of encoded sentiments
            #
            #---------------------------------------------------------
            tokenizedReviews = tokenizedReviews.to(device)
            encodedSentiments = encodedSentiments.to(device)
            
            #--------------------------------------------------------
            # Create Hidden State Variables
            #--------------------------------------------------------
            #
            # Create new variables for the hidden state, otherwise 
            # we'd be backprop through the entire training history
            #   -In effect this step removes the 
            #    grad_fn=<StackBackward0> calls at the end of each 
            #    semantic feature vector within h0 and c0; while 
            #    persisting the values in the vectors from the prior
            #    forward pass
            #   -Each grad_fn stored with the tensor allows you to 
            #    walk the computation all the way back to its inputs 
            #    with its next function property; drilling down on 
            #    this property shows the gradient functions for all 
            #    the prior tensors
            #
            # h starts as tuple of zeroed tensors (h0 for hidden 
            # state, c0 for cell state) of size no_layers (aka number 
            # of stacked LSTMs) x batch_size (aka data loader batch) x
            # hidden_dim (aka num semantic features in LSTM hidden 
            # state)
            #
            #---------------------------------------------------------
            h = tuple([each.data for each in h])
            
            #--------------------------------------------------------
            # Zero out model gradients
            #--------------------------------------------------------
            #
            # Set gradients of all model parameters to 0
            #
            # It is beneficial to zero out gradients when building
            # a NN.  This step clears old gradients from the last 
            # step, otherwise would just accumulate the gradients
            #
            #---------------------------------------------------------
            model.zero_grad()
            
            #--------------------------------------------------------
            # Perform forward pass
            #--------------------------------------------------------
            #
            # This step calls the forward method from the 
            # SentimentRNN class description which takes in a 
            # tokenizedReviews batch and the current LSTM hidden 
            # state, performs a forward pass through the NN 
            # architecture, and outputs the results from the sigmoid 
            # layer and the LSTM hidden state
            #
            # This step can take up to 1min per forward pass
            #
            #---------------------------------------------------------
            trainingSigOut, h = model(tokenizedReviews, h)
            
            #--------------------------------------------------------
            # Calculate the loss
            #--------------------------------------------------------
            #
            # Calculate the loss by measuring the binary cross 
            # entropy between:
            #   -trainingSigOut:
            #      +Sigmoid layer output of the model subsequent to 
            #       this forward pass
            #      +squeeze to remove single-dimension entries from 
            #       the shape of the model output
            #   -encodedSentiments:
            #      +Truth label for this batch
            #      +Convert to float to allow for comparison with 
            #       trainingSigOut (from int)
            #   -criterion returns loss as of class tensor
            #
            #---------------------------------------------------------
            loss = criterion(trainingSigOut.squeeze(), encodedSentiments.float())
            
            #--------------------------------------------------------
            # Perform backprop
            #--------------------------------------------------------
            #
            # Compute the gradient of the loss tensor during the 
            # backward pass:
            #   -Since the tensor is single-element, specifying 
            #    inputs not required
            #   -The .backward() method accumulates gradient in the 
            #    leaves; the gradients are stored in their 
            #    respective variables; gradients can be accessed via 
            #    the .grad attribute
            #   -On the .backward() call gradients are calculated 
            #    and stored for each model parameter's .grad 
            #    attribute; each model parameter (model.parameters()) 
            #    is a leaf tensor with the gradients stored in .grad
            #   -The model parameters for this model architecture 
            #    are:
            #      +embedding.weight
            #      +lstm.weight_ih_l0
            #      +lstm.weight_hh_l0
            #      +lstm.bias_ih_l0
            #      +lstm.bias_hh_l0
            #      +lstm.weight_ih_l1
            #      +lstm.weight_hh_l1
            #      +lstm.bias_ih_l1
            #      +lstm.bias_hh_l1
            #      +fc.weight
            #      +fc.bias
            #
            # Add loss value to the loss values list
            #   -The loss.item() returns the value as the underlying 
            #    data type, in this case float
            #
            # This step can take up to 8 seconds per gradient 
            # calculation
            #
            #---------------------------------------------------------
            loss.backward()
            
            #--------------------------------------------------------
            # Document Loss
            #--------------------------------------------------------
            #
            # Add loss value to the loss values list
            #   -The loss.item() returns the value as the underlying 
            #    data type, in this case float
            #
            #---------------------------------------------------------
            train_losses.append(loss.item())
            
            #---------------------------------------------------------
            # Calculate accuracy
            #---------------------------------------------------------
            #
            # Calculate summed comparison of True (1) and False (0) 
            # comparisons between model predicted sentiment and 
            # truth labeled sentiment
            #
            # Add accuracy sum from this round to the running 
            # training accuracy counter
            #
            #---------------------------------------------------------
            accuracy = acc(trainingSigOut, encodedSentiments)
            train_acc += accuracy
            
            #---------------------------------------------------------
            # Clip gradients
            #---------------------------------------------------------
            #
            # Clip the max norm of the gradients for the model's 
            # parameters to the clip value; this helps prevent the 
            # exploding gradient problem in RNNs / LSTM NNs
            # 
            # .clip_grad_norm_ method:
            #   -Clips the gradient norm of an iterable of parameters
            #   -The norm is computed over all gradients together as 
            #    if they were concatenated into a single vector
            #   -Gradeints are modified in place
            #
            #---------------------------------------------------------
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            
            #---------------------------------------------------------
            # Step the optimizer
            #---------------------------------------------------------
            # 
            # Update the model parameters based on the gradients
            #
            #---------------------------------------------------------
            optimizer.step()
            
            #--------------------------------------------------------
            # Increment counter
            #--------------------------------------------------------
            #
            # Increment counter for run time message
            #
            #---------------------------------------------------------
            train_loader_counter += 1
            
        #--------------------------------------------------------
        # User Information
        #--------------------------------------------------------
        #
        # Output run time user message
        #
        #---------------------------------------------------------
        logMessage = stringNow() + " Completed training data loader loop for Epoch " + str(epoch+1)     #user message
        print(logMessage)                                                                               #output message to screen
        lTrainingLog.append(logMessage + "\n")                                                          #capture message in log; append to string list
        
        #--------------------------------------------------------
        # Initialize validation loop objects
        #--------------------------------------------------------
        #
        # Initialize objects for this training epoch loop
        #
        #---------------------------------------------------------
        val_losses = []                                             #List to hold loss values during validation
        val_acc = 0.0                                               #Variable to hold model accuracies vs truth during validation
        val_h = model.init_hidden(batch_size)                       #Initialize the model's hidden state; returns tuple of zerod tensors
        valid_loader_counter = 0                                    #instantiate validation loader counter

        #--------------------------------------------------------
        # Set up model for validation
        #--------------------------------------------------------
        #
        # Set up model for the validation epoch loop
        #
        #---------------------------------------------------------
        model.eval()                                                #Set the model to evaluation mode
        
        #--------------------------------------------------------
        # User Information
        #--------------------------------------------------------
        #
        # Output validation run time user message
        #
        #---------------------------------------------------------
        logMessage = stringNow() + " Begining validation data loader loop Epoch " + str(epoch+1)                                #user message
        print(logMessage)                                                                                                       #output message to screen
        lTrainingLog.append(logMessage + "\n")                                                                                  #capture message in log; append to string list
        logMessage = "This step takes up to 30s per element, valid_loader elements to process: " + str(len(valid_loader))       #user message
        print(logMessage)                                                                                                       #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")                                                                         #capture message in log; append to string list
        logMessage = "Processing valid_loader elements: "                                                                       #user message
        print(logMessage)                                                                                                       #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")                                                                         #capture message in log; append to string list
        
        #--------------------------------------------------------
        # Validation data loop
        #--------------------------------------------------------
        #
        # Loop over tokenized reviews & encoded sentiments 
        # batches in the validation data loader
        #
        #---------------------------------------------------------
        for tokenizedReviews, encodedSentiments in train_loader:
            
            #--------------------------------------------------------
            # User Information
            #--------------------------------------------------------
            #
            # Output run time user message
            #
            #---------------------------------------------------------
            logMessage = stringNow() + " Processing element => " + str(valid_loader_counter)        #user message
            print(logMessage)                                                                       #output message to screen
            lTrainingLog.append(logMessage + "\n")                                                  #capture message in log; append to string list
                
            #--------------------------------------------------------
            # Use for debugging
            #--------------------------------------------------------
            #
            # This is error-checking code only:
            #   -break out of loop after completing 2 elements
            #   -use to check code in a reasonable amount of time
            #
            #---------------------------------------------------------
            if valid_loader_counter > 2:
               print(f"Debugging:  Breaking valid_loader loop at element: {valid_loader_counter}")
               break
                
            #--------------------------------------------------------
            # Move to device
            #--------------------------------------------------------
            #
            # Move training data tensors to device in anticipation of 
            # calculations:
            #   -this batch of tokenized reviews
            #   -this batch of encoded sentiments
            #
            #---------------------------------------------------------
            tokenizedReviews = tokenizedReviews.to(device)
            encodedSentiments = encodedSentiments.to(device)
            
            #--------------------------------------------------------
            # Create Hidden State Variables
            #--------------------------------------------------------
            #
            # Create new variables for the hidden state, otherwise 
            # we'd be backprop through the entire training history
            #   -In effect this step removes the 
            #    grad_fn=<StackBackward0> calls at the end of each 
            #    semantic feature vector within h0 and c0; while 
            #    persisting the values in the vectors from the prior
            #    forward pass
            #   -Each grad_fn stored with the tensor allows you to 
            #    walk the computation all the way back to its inputs 
            #    with its next function property; drilling down on 
            #    this property shows the gradient functions for all 
            #    the prior tensors
            #
            # val_h starts as tuple of zeroed tensors (h0 for hidden 
            # state, c0 for cell state) of size no_layers (aka number 
            # of stacked LSTMs) x batch_size (aka data loader batch) x
            # hidden_dim (aka num semantic features in LSTM hidden 
            # state)
            #
            #---------------------------------------------------------
            val_h = tuple([each.data for each in val_h])
            
            #--------------------------------------------------------
            # Perform forward pass
            #--------------------------------------------------------
            #
            # This step calls the forward method from the 
            # SentimentRNN class description which takes in a 
            # tokenizedReviews batch and the current LSTM hidden 
            # state, performs a forward pass through the NN 
            # architecture, and outputs the results from the sigmoid 
            # layer and the LSTM hidden state
            #
            # This step can take up to 2 seconds per forward pass
            #
            #---------------------------------------------------------
            validationOut, val_h = model(tokenizedReviews, val_h)
            
            #--------------------------------------------------------
            # Calculate the loss
            #--------------------------------------------------------
            #
            # Calculate the loss by measuring the binary cross 
            # entropy between:
            #   -validationOut:
            #      +Sigmoid layer output of the model subsequent to 
            #       this forward pass
            #      +squeeze to remove single-dimension entries from 
            #       the shape of the model output
            #   -encodedSentiments:
            #      +Truth label for this batch
            #      +Convert to float to allow for comparison with 
            #       trainingSigOut (from int)
            #   -criterion returns loss as of class tensor
            #
            #---------------------------------------------------------
            val_loss = criterion(validationOut.squeeze(), encodedSentiments.float())
            
            #--------------------------------------------------------
            # Document Loss
            #--------------------------------------------------------
            #
            # Add loss value to the loss values list
            #   -The loss.item() returns the value as the underlying 
            #    data type, in this case float
            #
            #---------------------------------------------------------
            val_losses.append(val_loss.item())
            
            #---------------------------------------------------------
            # Calculate accuracy
            #---------------------------------------------------------
            #
            # Calculate summed comparison of True (1) and False (0) 
            # comparisons between model predicted sentiment and 
            # truth labeled sentiment
            #
            # Add accuracy sum from this round to the running 
            # validation accuracy counter
            #
            #---------------------------------------------------------
            accuracy = acc(validationOut, encodedSentiments)
            val_acc += accuracy
            
            #--------------------------------------------------------
            # Increment counter
            #--------------------------------------------------------
            #
            # Increment counter for run time message
            #
            #---------------------------------------------------------
            valid_loader_counter += 1
        
        #--------------------------------------------------------
        # User Information
        #--------------------------------------------------------
        #
        # Output run time user message
        #
        #---------------------------------------------------------
        logMessage = stringNow() + " Completed validation data loader loop for Epoch " + str(epoch+1)       #user message
        print(logMessage)                                                                                   #output message to screen
        lTrainingLog.append(logMessage + "\n")                                                              #capture message in log; append to string list
        
        #--------------------------------------------------------
        # Calculate Metrics
        #--------------------------------------------------------
        #
        # Calculate current epoch losses and accuracies
        #
        #---------------------------------------------------------
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = train_acc / len(train_loader.dataset)
        epoch_val_acc = val_acc / len(valid_loader.dataset)
        epoch_tr_loss.append(epoch_train_loss)
        epoch_vl_loss.append(epoch_val_loss)
        epoch_tr_acc.append(epoch_train_acc)
        epoch_vl_acc.append(epoch_val_acc)
        
        #--------------------------------------------------------
        # User Information
        #--------------------------------------------------------
        #
        # Output Epoch completion information for user
        #
        #---------------------------------------------------------
        logMessage = "Stats for Epoch " + str(epoch+1)              #user message
        print(logMessage)                                           #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")             #capture message in log; append to string list
        logMessage = "train_loss: " + str(epoch_train_loss)         #user message
        print(logMessage)                                           #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")             #capture message in log; append to string list
        logMessage = "val_loss: " + str(epoch_val_loss)             #user message
        print(logMessage)                                           #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")             #capture message in log; append to string list
        logMessage = "train_accuracy: " + str(epoch_train_acc*100)  #user message
        print(logMessage)                                           #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")             #capture message in log; append to string list
        logMessage = "val_accuracy: " + str(epoch_val_acc*100)      #user message
        print(logMessage)                                           #output message to screen
        lTrainingLog.append(" "*16 + logMessage + "\n")             #capture message in log; append to string list
               
        #--------------------------------------------------------
        # Capture Model Information
        #--------------------------------------------------------
        #
        # If this is the lowest loss so far; caputure model info
        #
        #---------------------------------------------------------
        if epoch_val_loss <= valid_loss_min:                                                                                        #If this is the lowest validation loss...
            logMessage = "Validation loss decreased " + str(valid_loss_min) + " --> " + str(epoch_val_loss) + " Saving model"       #user message - capture decrease in validation loss
            print(logMessage)                                                                                                       #output message to screen
            lTrainingLog.append(" "*16 + logMessage + "\n")                                                                         #capture message in log; append to string list
            model.save()                                                                                                            #run model save method to save model state dictionary and vocab
            valid_loss_min = epoch_val_loss                                                                                         #update validation loss value
        
    #--------------------------------------------------------
    # User Information
    #--------------------------------------------------------
    #
    # Output training complete user message
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Model training complete"           #user message
    print(logMessage + "\n" + "-"*70)                               #output message to screen
    lTrainingLog.append(logMessage + "\n")                          #capture message in log; append to string list
    
    #--------------------------------------------------------
    # Output training log
    #--------------------------------------------------------
    #
    # Save training log and inform user
    #
    #---------------------------------------------------------
    sRunningPyFileName = os.path.basename(__file__)[:-3]                                                #Pull the name of this python file; remove .py file extension
    sTrainingLogPath = model.save_dir + '\\' + stringNow() + "_" + sRunningPyFileName + "_log.txt"      #build a string for the path to save the log file
    with open(sTrainingLogPath, "w") as fTrainingLog:                                                   #open file for write; will overwrite any existing content
        for logElement in lTrainingLog:                                                                 #loop thru elements in the training log list
            fTrainingLog.write(logElement)                                                              #write this log element to the log file
    fTrainingLog.close()                                                                                #close file
    
#------------------------------------------
# Load Model Function Definition
#------------------------------------------
#
# Function to re-create the model structure 
# with a new model instance and load its 
# state dictionary
#
# Inputs:   csvPath, path to the IMDB database .csv file
# 
# Outputs:  myNewModel, loaded model; object of class SentimentRNN
#           
#
#------------------------------------------
def loadModel(stateDictPath, vocabDictPath):
    
    #--------------------------------------------------------
    # Load Model
    #--------------------------------------------------------
    #
    # -Load vocablary
    # -Build architecture
    # -Load model state
    #
    #---------------------------------------------------------
    logMessage = stringNow() + " Loading vocab dictionary from " + vocabDictPath        #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(logMessage + "\n")                                              #capture message in log; append to string list
    with open(vocabDictPath, "rb") as f:                                                #open vocab dict pickle file for reading
        loaded_vocab = pickle.load(f)                                                   #load pickel file into variable
    myNewModel = buildModel(loaded_vocab)                                               #Instantiate new model instance and build out model architecture; pass in loaded vocab dict
    logMessage = stringNow() + " Loading State Dict from " + stateDictPath              #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(logMessage + "\n")                                              #capture message in log; append to string list
    myNewModel.load_state_dict(torch.load(stateDictPath))                               #load model state dict to recover trained model
    logMessage = stringNow() + " Model load complete"                                   #user message
    print(logMessage)                                                                   #output message to screen
    lTrainingLog.append(logMessage + "\n")                                              #capture message in log; append to string list
    
    #--------------------------------------------------------
    # Set Model Into Evaluation Mode
    #--------------------------------------------------------
    #
    # Some models have different behaviors in training vs 
    # evaluation mode; for instance models with dropout 
    # layers
    #
    #---------------------------------------------------------
    myNewModel.eval()
    
    #--------------------------------------------------------
    # Function Returns
    #--------------------------------------------------------
    #
    # Loaded trained sentiment analysis model
    #
    #---------------------------------------------------------
    return myNewModel