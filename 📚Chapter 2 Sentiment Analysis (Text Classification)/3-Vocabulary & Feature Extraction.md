In this blog, you’re going to learn how to represent a text as a vector. In order for you to do so, you first have to build vocabulary and that will allow you to encode any text or any tweet as an array of numbers.

## 📑 Table of Contents  

- [What is Sentiment Analysis](#What-is-Sentiment-Analysis)  
- [How does sentiment analysis work](#How-does-sentiment-analysis-work)  
- [Supervised Machine Learning](#Supervised-Machine-Learning)  
- [Logistic Regression and Sentiment Analysis?](#Logistic-Regression-and-Sentiment-Analysis?)
- [Challenges and Limitations](#Challenges-and-Limitations)   
- [Conclusion](#Conclusion)  


#  **Vocabulary** 

def:In the context of natural language processing (NLP), vocabulary refers to the set of unique words that appear in a text corpus.The vocabulary is used to represent text in a machine-readable format. For example, if the vocabulary contains 10,000 words, then each text document can be represented as a vector of 10,000 numbers, where each number represents the frequency of a particular word in the document.

The vocabulary is an important part of NLP because it allows us to represent text in a way that can be processed by computers. Without a vocabulary, it would be very difficult to develop NLP algorithms that can understand and process natural language.

There are a few different ways to create a vocabulary for NLP. One common approach is to use a **statistical method** to select the most frequent words in a text corpus. Another approach is to use a **dictionary** of words that are considered to be important for a particular task.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/ec4e5245-28c6-415d-b7b3-d17d7bbf4de4_700x232.jpg"></a>
</p>

So let’s dive in and see how you can do this. Picture a list of tweets, visually it would look like this. Then your vocabulary, V, would be the list of unique words from your list of tweets. To get that list, you’ll have to go through all the words from all your tweets and save every new word that appears in your search. So in this example, you’d have the word I, then the word, am and happy, because, and so forth. But note that the word I and the word am would not be repeated
in the vocabulary.

 # **Feature Extraction**
## Sparse Representations

Let’s take these tweets and extract features using your vocabulary. To do so, you’d have to check if every word from your vocabulary appears in the tweet. If it does like in the case of the word I, you would assign a value of 1 to that feature, like this. If it doesn’t appear, you’d assign a value of 0, like that.

In this example, the representation of your tweet would have six ones and many zeros. These correspond to every unique word from your vocabulary that isn’t in the tweet.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/1.jpg"></a>
</p>

Now, this type of representation with a small relative number of non-zero values is called a sparse representation. Now let’s take a closer look at this representation of these tweets. In the last slides, I walked you through extracting features to represent the tweet based on a vocabulary and I arrived at this vector.


<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/fd395a5c-5ea4-4a37-9a6a-f04df9241b82_700x268.jpg"></a>
</p>


This representation would have a number of features equal to the size of your entire vocabulary. This would have a lot of features equal to 0 for every tweet. With the sparse representation, a logistic regression model would have to learn n plus 1 parameters, where n would be equal to the size of your vocabulary and you can imagine that for large vocabulary sizes, this would be problematic. It would take an excessive amount of time to train your model and much more time than necessary to make predictions.

Given a text, you learned how to represent this text as a vector of dimension V. Specifically, you did this for a tweet and you were able to build a vocabulary of dimension V. Now as V gets larger and larger, you will face certain problems. In the next video, you will learn to identify these problems.



### References

1-[public pre-trained models for sentiment analysis on Hugging Face.](https://huggingface.co/models?search=sentiment)

2-[Two minutes NLP — Quick Intro to Sentiment Analysis](https://medium.com/nlplanet/two-minutes-nlp-quick-intro-to-sentiment-analysis-106b6947b2fd)

4-[Understanding the Emotion Tone of Text with AI — Sentiment Analysis on Monkeypox Tweets](https://pub.towardsai.net/understanding-the-emotion-tone-of-text-with-ai-sentiment-analysis-on-monkeypox-tweets-13040cfb1f99)

5-[Sentiment Analysis: Marketing with Large Language Models (LLMs)](https://medium.com/codex/computer-vision-fundamentals-with-opencv-9fc93b61e3e8)


<p align="right"><a target="_blank" href="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%201%20Introduction/What%20is%20NLP.md"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/home_page.png" /></a><a target="_blank" href="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%201%20Introduction/What%20is%20NLP.md"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/contents_page.jpg" /></a><a target="_blank" href="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%201%20Introduction/What%20is%20NLP.md"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/back_page.png" /></a><a target="_blank" href="https://coursesteach.com/mod/page/view.php?id=6320&amp;forceview=1"><img height="50px" src="https://raw.githubusercontent.com/dipanjanS/practical-machine-learning-with-python/master/media/assets/next_page.png" /></a></p>

### 🧠 Sentiment Analysis — Supervised Machine Learning Quiz

---

**1. In supervised machine learning for sentiment analysis, what is the primary role of the cost function?**

A. To categorize the final output into positive, negative, or neutral sentiment categories.  
B. To convert the raw text of a tweet into a format that can be used for binary classification.  
C. To select the best feature extraction technique like bag-of-words or TF-IDF.  
**D. To measure the difference between the model's predicted sentiments and the actual sentiments in the training data.**

---

**2. What is the primary function of the sigmoid (or logistic) function within a logistic regression model for sentiment analysis?**

A. To calculate evaluation metrics such as accuracy, precision, and recall.  
B. To split the dataset into separate training and testing sets.  
C. To clean and preprocess the raw text by removing irrelevant words.  
**D. To map the model's linear output to a probability value between 0 and 1.**

---

**3. According to the source material, which of the following is considered a significant challenge that can lead to misinterpretation in sentiment analysis?**

A. The difficulty in splitting data into training and testing sets.  
**B. The presence of sarcasm and irony in a piece of text.**  
C. The need to choose an appropriate programming language like Python.  
D. The large volume of data available on social media platforms.  

---

**4. What is the process of converting text into numerical features, such as using bag-of-words or TF-IDF, called in the context of sentiment analysis?**

**A. Feature Extraction**  
B. Sentiment Classification  
C. Model Evaluation  
D. Data Labeling  

---

**5. The provided text states that sentiment analysis is also known by another name. What is that name?**

A. Text Summarization  
**B. Emotion AI**  
C. Natural Language Generation  
D. Opinion Mining  

---

**6. In the process of building a sentiment classifier with logistic regression, what is the first step you would take with the raw tweets?**

A. Evaluate the model's performance using accuracy and F1 score.  
B. Use the sigmoid function to predict the probability of a positive sentiment.  
**C. Process the tweets to clean the text and label the data.**  
D. Train the logistic regression classifier while minimizing cost.  

---

**7. Which of the following is NOT described in the source material as a real-world application of sentiment analysis?**

A. Tracking the sentiment of a brand over time to adjust marketing efforts.  
**B. Correcting grammatical errors in customer feedback emails.**  
C. Monitoring employee morale to help managers identify problem areas.  
D. Automatically generating product recommendations based on user sentiment.  

---

**8. For a binary sentiment classification task using logistic regression, how are the training labels for 'positive' and 'negative' tweets typically represented?**

A. As a probability score between 0 and 1.  
B. As a vector of word embeddings.  
**C. As numerical values, such as 1 for positive and 0 for negative.**  
D. As the words 'positive' and 'negative'.  

---

















































