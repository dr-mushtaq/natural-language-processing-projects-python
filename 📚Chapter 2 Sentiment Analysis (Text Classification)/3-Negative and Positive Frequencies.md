# 📊 Understanding Positive and Negative Word Frequencies in NLP

A frequency dictionary is a specialized type of lexicon used in Natural Language Processing (NLP) that organizes words based on their frequency of occurrence within a specific corpus. This tool is particularly beneficial for linguistic research, language learning, and various applications in text analysis.

## 📑 Table of Contents  

- [Vocabulary in NLP](#Vocabulary-in-NLP)  
- [Feature Extraction](#Feature-Extraction)  
- [Simple Python Example](#Simple-Python-Example)  


#  **What Is a Frequency Dictionary in NLP?** 

A frequency dictionary in NLP is a list of all the unique words occurring in a corpus, along with their frequencies. The frequency of a word is the number of times it appears in the corpus. Frequency dictionaries are used in a variety of NLP tasks, such as:

A frequency dictionary is a special type of lexicon that organizes words based on how often they appear in a given text or dataset.
In Natural Language Processing (NLP), this tool helps researchers and developers understand the importance of specific words in a corpus.

Simply put — it’s a list of all unique words in your dataset, along with the number of times each word occurs.

These frequency counts become powerful features in various NLP tasks, from sentiment analysis to document classification.

# Key Applications of Frequency Dictionaries

Frequency dictionaries are used in multiple NLP applications, including:

- **Document classification:** Frequency dictionaries can be used to represent documents as vectors of word frequencies. This can then be used to train machine learning models to classify documents into different categories.

- **Sentiment analysis:** Frequency dictionaries can be used to identify the sentiment of a document. For example, a document that contains a lot of words with positive sentiment scores is likely to be positive, while a document that contains a lot of words with negative sentiment scores is likely to be negative.

- **Topic modeling:** Frequency dictionaries can be used to identify the topics of a document. This is done by clustering words together based on their frequencies. The clusters of words are then used to represent the topics of the document.

# How Frequency Dictionaries Are Built?

There are two main approaches:

- **Manual creation:** Counting word occurrences by hand — often done for small datasets or linguistic studies.
- **Automatic generation:** Using NLP libraries (like NLTK, scikit-learn, or spaCy) to calculate word frequencies statistically across large corpora.
- 
Most modern applications rely on automatic methods since they’re faster and more scalable.

# Using Frequency Dictionaries in Logistic Regression

Frequency dictionaries can serve as **feature generators** for machine learning models.
For example, in sentiment classification, you can track:

- How often a word appears in **positive** documents
- How often it appears in **negative** documents

These two counts — known as **positive and negative frequencies** — can then be used as input features for a logistic regression model that predicts sentiment polarity.

# 😊 Example: Positive and Negative Word Frequencies

Imagine a small corpus of four tweets:

- Two belong to the positive class
- Two belong to the negative class

From these tweets, you build a vocabulary of all unique words.
Then, you count how many times each word appears in positive versus negative tweets.

For example:'

- The word “happy” appears twice in positive tweets → Positive Frequency = 2
- The word “am” appears three times in negative tweets → Negative Frequency = 3

These counts are stored in a dictionary structure, mapping each word and class to its frequency.

In practice, you end up with something like:

So let’s take a look at how you can do that. It is helpful to first imagine how these two classes would look.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/6d091ab7-9d0f-46ea-bd70-9de7ca9c7c92_700x278.jpg"></a>
</p>

Here for instance, you could have a corpus consisting of four tweets. Associated with that corpus, you would have a set of unique words, your vocabulary. In this example, your vocabulary would have eight unique words.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/2abc1674-c1eb-4281-9df4-7d9aade17cb7_700x125.jpg"></a>
</p>


For this particular example of sentiment analysis, you have two classes. One class is associated with positive sentiment and the other with negative sentiment. So taking your corpus, you’d have a set of two tweets that belong to the positive class,and the sets of two tweets that belong to the negative class.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/d16b22bf-67db-418d-8d00-0889bd210bd0_700x298.jpg"></a>
</p>


Let’s take the sets of positive tweets. Now, take a look at your vocabulary. To get the positive frequency in any word in your vocabulary, you will have to count the times as it appears in the positive tweets. For instance, the word happy appears one time in the first positive tweet, and another time in the second positive tweet. So it’s positive frequency is two. The complete table looks like this. Feel free to take a pause and check any of its entries.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/download%20(3).png"></a>
</p>


The same logic applies for getting the negative frequency. However, for the sake of clarity, look at some examples, the word am appears two times in the first tweet and another time in the second one. So it’s negative frequency is three. Take a look at the entire table for negative frequencies and feel free to check its values.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/download%20(4).png"></a>
</p>


So this is the entire table with the positive and negative frequencies for your corpus. In practice when coding, this table is a dictionary mapping from a word class there to its frequency. So it maps the word and its corresponding class to the frequency or the number of times that’s where it showed up in the class. You now know how to create a frequency dictionary, which maps a word and the class to the number of times that word showed up in the corresponding class.


### References














