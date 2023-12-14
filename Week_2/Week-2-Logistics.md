# Basics of Machine Learning

This week we will start by learning about Logistic Regression Models and Data Preprocessing. 

Understanding the basics of simple machine learning models is crucial for our initial steps toward their implementation. Preprocessing plays a vital role in ensuring the accuracy and reliability of our data.

Using online python environments like [Google Colab](https://colab.research.google.com/) is encouraged.

## Introduction

Machine learning is a subfield of artificial intelligence that involves the development of algorithms and statistical models that enable computers to improve their performance in tasks through experience. These algorithms and models are designed to learn from data and make predictions or decisions without explicit instructions. Machine learning is used in a wide variety of applications, including image and speech recognition, natural language processing (our project), and recommender systems. 

### Classification of Machine Learning Problems
Machine learning implementations are classified into three major categories:

1. **Supervised Learning** : Supervised learning is the machine learning task of learning a function that maps an input to an output based on example input-output pairs. The given data is labeled. Both classification (This is what we want to do) and regression problems are supervised learning problems. In classification, we want our model to look at features, e.g., text in a review, and then predict to which category (sometimes called a class) among some discrete set of options, an example belongs. The simplest classification problem is an Binary Classification Problem (basically a Yes/No or 0/1 problem). Some applications include Spam Filteration, Disease Diagnosis and Hate Speech Recogntion. When labels take on arbitrary numerical values (even within some interval), we call it a regression. The goal is to produce a model whose predictions closely approximate the actual label values. Basically regression problems answers the question **How much/many?**. One such application is predicting house prices based on input features like size and location.

2. **Unsupervised and Self Supervised Learning** : Unsupervised learning is a type of machine learning algorithm used to draw inferences from datasets consisting of input data without labeled responses. In unsupervised learning algorithms, classification or categorization is not included in the observations. As a kind of learning, it resembles the methods humans use to figure out that certain objects or events are from the same class, such as by observing the degree of similarity between objects. Some recommendation systems that you find on the web in the form of marketing automation are based on this type of learning.

3. **Reinforcement Learning** : Reinforcement learning is the problem of getting an agent to act in the world so as to maximize its rewards. A learner is not told what actions to take as in most forms of machine learning but instead must discover which actions yield the most reward by trying them. For example — Consider teaching a dog a new trick: we cannot tell him what to do, what not to do, but we can reward/punish it if it does the right/wrong thing.

### Basic Terminologies 

**Model** : A model is a specific representation learned from data by applying some machine learning algorithm.

**Feature** : A feature is an individual measurable property of our data. A set of numeric features can be conveniently described by a feature vector. Feature vectors are fed as input to the model. For example, in order to predict a fruit, there may be features like color, smell, taste, etc. Note: Choosing informative, discriminating and independent features is a crucial step for effective algorithms. We generally employ a feature extractor to extract the relevant features from the raw data.

**Target (Label)** : A target variable or label is the value to be predicted by our model. For the fruit example discussed in the features section, the label with each set of input would be the name of the fruit like apple, orange, banana, etc.

**Training** : The idea is to give a set of inputs(features) and its expected outputs(labels), so after training, we will have a model that will then map new data to one of the categories trained on. We will see how it does this.

**Prediction** : Once our model is ready, it can be fed a set of inputs to which it will provide a predicted output(label). But make sure if the machine performs well on unseen data, then only we can say the machine performs well.

### Steps 
1. **Define the Problem**: Identify the problem you want to solve and determine if machine learning can be used to solve it.
2. **Collect Data**: Gather and clean the data that you will use to train your model. The quality of your model will depend on the quality of your data.
3. **Explore the Data**: Use data visualization and statistical methods to understand the structure and relationships within your data.
4. **Pre-process the Data**: Prepare the data for modeling by normalizing, transforming, and cleaning it as necessary.
5. **Split the Data**: Divide the data into training and test datasets to validate your model.
6. **Choose a Model**: Select a machine learning model that is appropriate for your problem and the data you have collected.
7. **Train the Model**: Use the training data to train the model, adjusting its parameters to fit the data as accurately as possible.
8. **Evaluate the Model**: Use the test data to evaluate the performance of the model and determine its accuracy.
9. **Fine-tune the Model**: Based on the results of the evaluation, fine-tune the model by adjusting its parameters and repeating the training process until the desired level of accuracy is achieved.
10. **Deploy the Model**: Integrate the model into your application or system.

## Content
This week has the following content: 

## Data Preprocessing & Visualisation 

Data preprocessing is an essential step in building a Machine Learning model and depending on how well the data has been preprocessed, the results are seen. In NLP (Natural Language Processing), text preprocessing is the first step in the process of building a model. The various text preprocessing steps are :

1. Text Cleaning
2. Stop words removal
3. Tokenization
4. Stemming & Lemmatization

Go through this [Colab](https://colab.research.google.com/drive/1fjZed668JZQ3q8q_27wZ0Dtp159g2TDY?usp=sharing). 

### Text Cleaning
In this step, we will perform fundamental actions to clean the text. These actions involve transforming all the text to lowercase, eliminating characters that do not qualify as words or whitespace, as well as removing any numerical digits present.

1. **Converting to lowercase**

Python is a case sensitive programming language. Therefore, to avoid any issues and ensure consistency in the processing of the text, we convert all the text to lowercase.
This way, “LOVE” and “love” will be treated as the same word, and our data analysis will be more accurate and reliable.

```python
for columns in dataset.columns:
    dataset[columns] = dataset[columns].str.lower()
```

2. **Removing URLs**

When building a model, URLs are usually not relevant and can be removed from the text data.

```python
import pandas as pd
import re

# Define a regex pattern to match URLs
url_pattern = re.compile(r'https?://\S+')

# Define a function to remove URLs from text
def remove_urls(text):
    return url_pattern.sub('', text)

# Apply the function to the 'text' column and create a new column 'clean_text'
df['Message'] = df['Message'].apply(remove_urls)
```

3. **Removing remove non-word and non-whitespace characters**

It is essential to remove any characters that are not considered as words or whitespace from the text dataset. These non-word and non-whitespace characters can include punctuation marks, symbols, and other special characters that do not provide any meaningful information for our analysis.

```python
df = df.replace(to_replace=r'[^\w\s]', value='', regex=True)
```

4. **Removing digits**

It is important to remove all numerical digits from the text dataset. This is because, in most cases, numerical values do not provide any significant meaning to the text analysis process. Moreover, they can interfere with natural language processing algorithms, which are designed to understand and process text-based information.

```python
df = df.replace(to_replace=r'\d', value='', regex=True)
```

### Tokenization
Tokenization is the process of breaking down large blocks of text such as paragraphs and sentences into smaller, more manageable units called tokens which can more easily assigned meaning. Tokens can be either words, characters, or subwords. Hence, tokenization can be broadly classified into 3 types – word, character, and subword (n-gram characters) tokenization. By performing word tokenization, we can obtain a more accurate representation of the underlying patterns and trends present in the text data. In above colab we performed Word Tokenisation.

```python
import nltk
from nltk.tokenize import word_tokenize

df['Message'] = df['Message'].apply(word_tokenize)
```

### Stopword Removal
Stopwords refer to the most commonly occurring words in any natural language. For the purpose of analyzing text data and building NLP models, these stopwords might not add much value to the meaning of the document. Therefore, removing stopwords can help us to focus on the most important information in the text and improve the accuracy of our analysis. One of the advantages of removing stopwords is that it can reduce the size of the dataset, which in turn reduces the training time required for natural language processing models. We used the NLTK library to remove stopwords from our dataset.

```python
import nltk
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))
df['Message'] = df['Message'].apply(lambda x: [word for word in x if word not in stop_words])
```

### Stemming & Lemmatization
Stemming is transforming any word to its most general form or base stem, which can be defined as a set of characters that are used to construct the word and its derivatives. Eg. automate, automatic, automation gives automat. However, in some cases, stemming process produces words that are not correct spellings of the root word. For e.g., we can look at the set of words that comprises the different forms of happy: happy, happiness and happier. We can see that the prefix happi the most common stem throughout the entire set of related words. Note that we cannot choose happ because it is the stem of unrelated words like happen. [Porter Stemming Algorithm](https://tartarus.org/martin/PorterStemmer/) is a common algorithm for carrying out stemming in popular NLP libraries like NLTK.

While Lemmatization aims to remove inflectional endings only and to return base or dictionary form of a word, which is known as lemma. Eg. are, is, am gives be. 

```python
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd

# Initialize the Porter Stemmer
stemmer = PorterStemmer()

# Define a function to perform stemming on the 'text' column
def stem_words(words):
    return [stemmer.stem(word) for word in words]

# Apply the function to the 'text' column and create a new column 'stemmed_text'
df['stemmed_messages'] = df['Message'].apply(stem_words)
```

```python
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
  words = text.split()
  return [lemmatizer.lemmatize(word) for word in words]
  
df['text'] = df['text'].apply(lemmatize_words)
```

Note that, we only use either Stemming or Lemmatization on our dataset based on the requirement.

## Logistic Regression








