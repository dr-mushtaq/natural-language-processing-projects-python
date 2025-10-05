Sentiment analysis, also known as opinion mining, is a technique used to determine the sentiment or emotion expressed in a piece of text. It has gained significant popularity in recent years due to the rise of social media and the need to understand customer opinions and feedback. In this blog post, we will explore how to perform sentiment analysis using logistic regression with Python.


## 📑 Table of Contents  

- [What is Sentiment Analysis](#What-is-Sentiment-Analysis)  
- [How does sentiment analysis work](#How-does-sentiment-analysis-work)  
- [What is Computer Vision NOT?](#what-is-computer-vision-not)  
- [How does Computer Vision work?](#3-how-does-computer-vision-work)
- [Real life Example?](#Real-life-Example)   
- [History of Computer Vision](#history-of-computer-vision)  


### **What is Sentiment Analysis** 
Sentiment analysis is a natural language processing (NLP) technique that aims to understand and categorize the sentiment expressed in a given text. It involves analyzing the words, phrases, and context of the text to determine whether the sentiment is positive, negative, or neutral.

Sentiment analysis, also known as opinion mining, is a technique used to determine the sentiment or emotion expressed in a piece of text. It has gained significant popularity in recent years due to its applications in various fields such as marketing, customer feedback analysis, and social media monitoring. In this blog post, we will explore the concept of sentiment analysis and delve into the details of using logistic regression as a powerful tool for sentiment classification.

In this section, we will provide an overview of sentiment analysis and its importance in today’s data-driven world.

Sentiment analysis is the process of extracting subjective information from text and determining the sentiment or emotion associated with it. It involves analyzing the text to classify it into positive, negative, or neutral sentiment categories. The main goal of sentiment analysis is to understand the opinions, attitudes, and emotions expressed by individuals or groups. This information can then be used to make informed decisions, improve customer service, and gain valuable insights.

**Def:** Sentiment analysis has been widely used since the early 20th century, and its research area is still fast growing. One of the most advanced solutions is to use AI to proceed with sentiment analysis. The algorithm uses a natural language processing (NLP) technique which enables it to determine the moods or emotions of a piece of text. In this case, companies can react based on user feedback.

**Def:** Sentiment analysis is a Natural Language Processing (NLP) [2] technique used to determine the sentiment of a text by automatically identifying its underlying opinions. The sentiment can be positive (e.g. “I’m very happy today”), negative (e.g. “I didn’t like that movie”), or neutral (e.g. “Today is Friday”, which may be subjectively seen as positive by some people actually 😁) [1]

 ## **How does sentiment analysis work**
 
Sentiment analysis typically works by first identifying the sentiment of individual words or phrases. This can be done using a variety of methods, such as **lexicon-based analysis**, **machine learning**, or **natural language processing**.

Once the sentiment of individual words or phrases has been identified, they can be combined to determine the overall feeling of a piece of text. This can be done using a variety of techniques, such as **sentiment scoring** or **sentiment classification**.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/1.jpg"></a>
</p>

Background: machine-learning classification task of sentiment analysis. In this example you have the tweet, let’s say, I’m happy because I’m learning NLP. And the objective in this task is to predict whether a tweet has a positive or negative sentiment. And you’ll do this by starting with a training set where tweets with a positive sentiment have a label of one, and the tweets with a negative sentiment have a label of zero.


In order for you to implement logistic regression, you need to take a few steps. In this tutorial you will learn about the steps required in order to implement this algorithm, so let’s take a look.

### **Supervised Machine Learning** 

In supervised machine learning, you have input features X and a set of labels Y. Now to make sure you’re getting the most accurate predictions based on your data, your goal is to minimize your error rates or cost as much as possible. And to do this, you’re going to run your prediction function which takes in parameters data to map your features to output labels Y hat.

Now the best mapping from features to labels is achieved when the difference between the expected values Y and the predicted values Y hat is minimized. Which the cost function does by comparing how closely your output Y hat is to your label Y. Then you can update your parameters and repeat the whole process until your cost is minimized. So let’s take a look at the supervised

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/3.jpg"></a>
</p>
 
 
###  **Logistic Regression and Sentiment Analysis**?

Logistic regression is a statistical model used to predict binary outcomes. It is commonly used when the dependent variable is dichotomous, meaning it can take only two values. In the context of sentiment analysis, the binary outcome represents the sentiment category (positive or negative).

Logistic regression is a popular machine learning algorithm used for binary classification problems. It is well-suited for sentiment analysis because it can handle text data and provide probabilistic outputs. Logistic regression models are interpretable and can capture nonlinear relationships between features and labels.

Logistic regression works by estimating the probability of an event occurring based on a set of independent variables. It uses a logistic function (also known as a sigmoid function) to map the linear combination of the independent variables to a value between 0 and 1. This value represents the probability of the event occurring.

For this task you will be using your logistic regression classifier which assigns its observations to two distinct classes. Next up I’ll show you how to do this. So to get started building a logistic regression classifier that’s capable of predicting sentiments of an arbitrary tweet.


<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/2.jpg"></a>
</p>

You will first process the raw tweets in your training sets and extract useful features. Then you will train your logistic regression classifier while minimizing the cost. And finally you’ll be able to make your predictions. So in this blog you learned about the steps required for you to classify a tweet. Given the tweet, you should classify it to either be positive or negative. In order for you to do so, you first have to extract the features. Then you have to train your model. And then you have to classify the tweet based off your trained model. In the next video, you’re going to learn how to extract these features. So let’s take a look at how you can do that

### References

1-[What is Computer Vision? & Its Applications](https://medium.com/@draj0718/what-is-computer-vision-its-applications-826c0bbd772b)

2-[-Introduction of Computer Vision](https://auth.udacity.com/sign-in)

4-[How computer vision works](https://www.sas.com/en_us/insights/analytics/computer-vision.html#technical)

5-[Computer Vision 🤖 Fundamentals with OpenCV](https://medium.com/codex/computer-vision-fundamentals-with-opencv-9fc93b61e3e8)


<p align="right"><a target="_blank" href="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%201%20Introduction/What%20is%20NLP.md"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/home_page.png" /></a><a target="_blank" href="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%201%20Introduction/What%20is%20NLP.md"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/contents_page.jpg" /></a><a target="_blank" href="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%201%20Introduction/What%20is%20NLP.md"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/back_page.png" /></a><a target="_blank" href="https://coursesteach.com/mod/page/view.php?id=6320&amp;forceview=1"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/next_page.png" /></a></p>

































