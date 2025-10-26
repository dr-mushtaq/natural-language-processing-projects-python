A frequency dictionary is a specialized type of lexicon used in Natural Language Processing (NLP) that organizes words based on their frequency of occurrence within a specific corpus. This tool is particularly beneficial for linguistic research, language learning, and various applications in text analysis.

## 📑 Table of Contents  

- [Vocabulary in NLP](#Vocabulary-in-NLP)  
- [Feature Extraction](#Feature-Extraction)  
- [Simple Python Example](#Simple-Python-Example)  


#  **What is the frequency dictionary (Bar)** 

A frequency dictionary in NLP is a list of all the unique words occurring in a corpus, along with their frequencies. The frequency of a word is the number of times it appears in the corpus. Frequency dictionaries are used in a variety of NLP tasks, such as:

# Key Applications of Frequency Dictionaries in NLP

- **Document classification:** Frequency dictionaries can be used to represent documents as vectors of word frequencies. This can then be used to train machine learning models to classify documents into different categories.

- **Sentiment analysis:** Frequency dictionaries can be used to identify the sentiment of a document. For example, a document that contains a lot of words with positive sentiment scores is likely to be positive, while a document that contains a lot of words with negative sentiment scores is likely to be negative.

- **Topic modeling:** Frequency dictionaries can be used to identify the topics of a document. This is done by clustering words together based on their frequencies. The clusters of words are then used to represent the topics of the document.

# How Are Frequency Dictionaries Created?

- Frequency dictionaries can be created manually or automatically. Manually created frequency dictionaries are created by counting the frequency of each word in a corpus.
- Automatic frequency dictionaries are created using statistical techniques.

# Using Frequency Dictionaries for Logistic Regression

We’ll now learn to generate counts, which you can then use as features in your logistic regression classifier. Specifically, given a word, you want to keep track of the number of times, that’s where it shows up as the positive class. Given another word you want to keep track of the number of times that word showed up in the negative class. Using both those counts, you can then extract features and use those features into your logistic regression classifier.

# Example: Positive and Negative Word Frequencies

So let’s take a look at how you can do that. It is helpful to first imagine how these two classes would look.


Here for instance, you could have a corpus consisting of four tweets. Associated with that corpus, you would have a set of unique words, your vocabulary. In this example, your vocabulary would have eight unique words.


For this particular example of sentiment analysis, you have two classes. One class is associated with positive sentiment and the other with negative sentiment. So taking your corpus, you’d have a set of two tweets that belong to the positive class,and the sets of two tweets that belong to the negative class.


Let’s take the sets of positive tweets. Now, take a look at your vocabulary. To get the positive frequency in any word in your vocabulary, you will have to count the times as it appears in the positive tweets. For instance, the word happy appears one time in the first positive tweet, and another time in the second positive tweet. So it’s positive frequency is two. The complete table looks like this. Feel free to take a pause and check any of its entries.


The same logic applies for getting the negative frequency. However, for the sake of clarity, look at some examples, the word am appears two times in the first tweet and another time in the second one. So it’s negative frequency is three. Take a look at the entire table for negative frequencies and feel free to check its values.


So this is the entire table with the positive and negative frequencies for your corpus. In practice when coding, this table is a dictionary mapping from a word class there to its frequency. So it maps the word and its corresponding class to the frequency or the number of times that’s where it showed up in the class. You now know how to create a frequency dictionary, which maps a word and the class to the number of times that word showed up in the corresponding class.


### References







