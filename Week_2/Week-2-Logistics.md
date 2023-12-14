# Basics of Machine Learning

This week we will start by learning about Logistic Regression Models, Data Preprocessing and Word Vectorisation. 

Understanding the basics of simple machine learning models is crucial for our initial steps toward their implementation. Preprocessing plays a vital role in ensuring the accuracy and reliability of our data. In this project, our goal is to create a sentiment analysis model, and one key aspect is Word Vectorization, which allows us to transform text data into numerical data.

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

## Data Preprocessing & Visualisation and Word Vectorisation
## Linear and Logistic Regression








