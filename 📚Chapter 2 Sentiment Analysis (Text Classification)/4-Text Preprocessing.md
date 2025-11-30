# Introduction

Data preparation is a critical stage in the data analysis process, ensuring that raw data is transformed into a clean and structured format suitable for analysis. Here are the essential steps involved in effective data preparation:

# Sections
- What is text-processing
- Main Steps involved in Data Preparation for Sentiment Analysis
- Detail discussion on steps involved in Data Preparation for Sentiment Analysis

# Section 1- What is text-processing

Def: Text pre-processing is the process of transforming unstructured text to structured text to prepare it for analysis.


<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/31.jpg"></a>
</p>

# Section 2- Why Text Preprocessing

-When you pre-process text before feeding it to algorithms, you increase the accuracy and efficiency of said algorithms by removing noise and other inconsistencies in the text that can make it hard for the computer to understand. 
-Making the text easier to understand also helps to reduce the time and resources required for the computer to pre-process data. These words need to then be encoded as integers, or floating-point values, for use as inputs in machine learning algorithms. This process is called feature extraction (or vectorizations).

Scikit-learn’s CountVectorizer is used to convert a collection of text documents to a vector of term/token counts. It also enables the ​pre-processing of text data prior to generating the vector representation. This functionality makes it a highly flexible feature representation module for text.

Take a look at the pictures below depicting two scenarios of an office space — one is untidy and the other is clean and organized [11].

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/32.jpg"></a>
</p>

You are searching for a document in this office space. In which scenario are you more likely to find the document easily? Of course, in the less cluttered one because each item is kept in its proper place. The data cleaning exercise is quite similar. If the data is arranged in a structured format then it becomes easier to find the right information[11].

The preprocessing of the text data is an essential step as it makes the raw text ready for mining, i.e., it becomes easier to extract information from the text and apply machine learning algorithms to it. If we skip this step then there is a higher chance that you are working with noisy and inconsistent data. The objective of this step is to clean noise those are less relevant to find the sentiment of tweets such as punctuation, special characters, numbers, and terms which don’t carry much weightage in context to the text[11].

In one of the later stages, we will be extracting numeric features from our Twitter text data. This feature space is created using all the unique words present in the entire data. So, if we preprocess our data well, then we would be able to get a better quality feature space.

# Section 3- Main Steps involved in Data Preparation for Sentiment Analysis
In this section, we will discuss the steps involved in preparing the data for sentiment analysis using logistic regression.

## 1- Data Collection:

The first step in any sentiment analysis project is to collect a suitable dataset. This can be done by scraping data from social media platforms, online reviews, or any other relevant sources.The first step for any supervised machine learning project is to gather the data to train and test your model.

Key considerations during this phase include:

- Identifying relevant data sources: Ensure the sources align with your analysis goals.
- Ensuring data accessibility: Confirm that you have the necessary permissions to access the data.
- Volume and freshness: Determine the appropriate amount of data needed and its update frequency

## 2- Data Cleaning

Once the data is collected, it needs to be cleaned by **removing unnecessary characters, punctuation marks, and stopwords. Stop-words** are words that do not carry much meaning and can be safely ignored. When it comes to low-level text processing problems… it is advisable to remove all punctuations and special characters ( including emojis…) for several significant reasons, which are Dimensionality issues, Computational Efficiency, Noise Reduction, and Generalization you can state other issues as well…

**Dimensionality Reduction:** Keeping every punctuation mark and special character as a separate feature can significantly increase the dimensionality of the data, making it computationally expensive and potentially leading to overfitting. By removing them, you reduce the dimensionality of the feature space.

**Computational Efficiency:** Some NLP algorithms and models, especially those based on neural networks, are computationally more efficient when trained on preprocessed text. Removing punctuations and special characters can help speed up the training and inference processes.

**Noise Reduction:** Punctuation and special characters often don’t carry significant semantic meaning on their own. Removing them can help reduce the noise in the text and make it easier for NLP models to focus on the meaningful words and phrases.

**Generalization:** Ignoring punctuations and special characters helps NLP models generalize better. For instance, if you remove the period from the end of a sentence, the model can better learn the relationship between words without being overly influenced by sentence boundaries.

However, it’s important to note that there are cases where punctuations and special characters might convey valuable information, such as in sentiment analysis (e.g., “I love it” vs. “I love it!”, in the second case the speaker seems to be more excited). In such cases, you may choose to retain certain punctuation marks or handle them differently in your preprocessing pipeline. The choice of whether to remove or retain punctuations depends on the specific NLP task and the goals of your analysis.

## 3- Tokenization:

**Def:** Tokenization is the process of splitting text into individual words or tokens. This step helps in creating a structured format for further analysis.

**Def:** To use text data for predicting stuff, you gotta break it down and get rid of some words — that’s called tokenization.

## 4- Feature Extraction:

**Def:** After tokenization, relevant features need to be extracted from the text. This can be done using techniques such as bag-of-words, TF-IDF (Term Frequency-Inverse Document Frequency), or word embeddings.

**Def:** To analyze preprocessed data, it needs to be converted into features. Depending upon the usage, text features can be constructed using assorted techniques — Bag of Words, TF IDF (Term Frequency — Inverse Document Frequency), Word2Vec (by Google), GloVe (Global Vectors by Stanford), FastText (by Facebook), ELMo (Embeddings from Language Models),GPT (Generative Pre-trained Transformer by OpenAI) BERT (Bidirectional Encoder Representations from Transformer by Google),LLM’s[11].

Def: Then you gotta turn those words into numbers, either integers or floating-point values, so you can use them in machine learning. That whole thing is known as feature extraction (or vectorization).

Labeling: Each data point in the dataset needs to be labeled with the corresponding sentiment category (positive or negative). This can be done manually or by using pre-labeled datasets for training purposes.

# Section 2- Detail discussion on steps involved in Data Preparation for Sentiment Analysis

## 1- Removing Punctuations, Numbers, and Special Characters (Remove hyperlinks, Twitter marks, and styles)

During human conversations, punctuation marks like ‘’, ! , [, }, *, #, /, ?, and ‘’ are incredibly relevant and necessary to have a proper conversation. They help to fully convey the message of the writer.

But in nlp , punctuations, numbers and special characters do not help much. It is better to remove them from the text just as we removed the twitter handles [11]

Since we have a Twitter dataset, we'd like to remove some substrings commonly used on the platform like the hashtag, retweet marks, and hyperlinks. We'll use the re library to perform regular expression operations on our tweet. We'll define our search pattern and use the sub() method to remove matches by substituting with an empty character (i.e. '

By removing punctuation marks from our text we allow the model to focus on the text alone rather than distracting it with symbols. This makes it easier for the text to be analysed.

<pre>
print('\033[92m' + tweet)
print('\033[94m')
# remove old style retweet text "RT"
tweet2 = re.sub(r'^RT[\s]+', '', tweet)
# remove hyperlinks
tweet2 = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet2)
# remove hashtags
# only removing the hash # sign from the word
tweet2 = re.sub(r'#', '', tweet2)
print(tweet2)

My beautiful sunflowers on a sunny Friday morning off :) #sunflowers #favourites #happy #Friday off… https://t.co/3tfYom0N1i

My beautiful sunflowers on a sunny Friday morning off :) sunflowers favourites happy Friday off…
  </pre>
  
## 2- Tokenization

**What is tokenization?**

Def: Tokenization is the process of transforming a string or document into smaller chunks, which we call tokens. When a sentence breakup into small individual words or Phrases, these pieces of words are known as tokens, and the process is known as tokenization. This is usually one step in the process of preparing a text for natural language processing.

Def: The tokenization process takes a body of text and breaks it into pieces, or tokens, which can then be transformed into a format the model can work with.

**Why is it Important?**

The model is only capable of processing numerical data, hence tokens are transformed into numbers. This step is crucial as it converts unstructured content into a format that the models can comprehend and evaluate. In the absence of tokenization, there would be no method to input text into machine learning models, since they can only interpret numerical values, not direct text strings.

Description : This allows the computer to work on your text token by token rather than working on the entire text in the following stage. There are many theories and rules regarding tokenization, and you can create your own tokenization rules using regular expressions, but normally tokenization will do things like break out words or sentences, often separate punctuation or you can even just tokenize parts of a string like separating all hashtags in a Tweet [1].

**Python Libraries**

- In spaCy, you can do either sentence tokenization or word tokenization
- The NLTK library includes a range of tokenizers for different languages and use cases.

**Why tokenize?**

Why bother with tokenization? Because it can help us with some simple text processing tasks like mapping part of speech, matching common words and perhaps removing unwanted tokens like common words or repeated words. Here, we have a good example. The sentence is: I don’t like Sam’s shoes. When we tokenize it we can clearly see the negation in the not and we can see possession with the ‘s. These indicators can help us determine meaning from simple text [1].

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/image%20(3).jpg"></a>
</p>

**Types of tokenization**

The two main types of tokenization are word and sentence tokenization.

1- **Word tokenization** is the most common kind of tokenization. Here, each token is a word, meaning the algorithm breaks down the entire text into individual words. breaks text down into individual words.

2- **sentence tokenization:** On the other hand, sentence tokenization breaks down text into sentences instead of words. It is a less common type of tokenisation only used in few Natural Language Processing (NLP) tasks.

<pre>
import spacy
text = """
... Dave watched as the forest burned up on the hill,
... only a few miles from his house. The car had
... been hastily packed and Marta was inside trying to round
... up the last of the pets. "Where could she be?" he wondered
... as he continued to wait for Marta to appear with the pets.
... """

nlp = spacy.load("en_core_web_sm")
doc = nlp(text)
token_list = [token for token in doc]
token_list
[
, Dave, watched, as, the, forest, burned, up, on, the, hill, ,,
, only, a, few, miles, from, his, house, ., The, car, had,
, been, hastily, packed, and, Marta, was, inside, trying, to, round,
, up, the, last, of, the, pets, ., ", Where, could, she, be, ?, ", he, wondered,
, as, he, continued, to, wait, for, Marta, to, appear, with, the, pets, .,
]

</pre>

In this code, you set up some example text to tokenize, load spaCy’s English model, and then tokenize the text by passing it into the nlp constructor. This model includes a default processing pipeline that you can customize[13]

# Tokenization algorithms

- whitespace tokenization,
- Regular expression tokenization
- statistical tokenization

# 3- Normalization

In normalization, your text is converted to standard form. An example of this is converting all text to lowercase, removing numbers, or removing punctuations. Normalization helps to make the text more consistent. There are a couple of different normalization techniques, but I’ll give you an explanation of some of the most commonly employed normalisation techniques below.

## Case normalization

This technique converts all the letters in your text to a single case, either uppercase or lowercase. Case normalisation ensures that your data is stored in a consistent format and makes it easier to work with the data. An example would be looking for all the instances of a word and searching for it in your text. Without case normalisation, the result of searching for the word ‘Boy’ would be different from the result of searching for ‘boy’.

## Stop words
Stop word is used to filter some words which are repeat often and not giving information about the text. In Spacy, there is a built-in list of some stop words.One of the important preprocessing steps in NLP is to remove stop words from text. Stop words are basically connector words such as ‘to’, ‘with’, ‘is’, etc. which provide minimal context. spaCy allows easy identification of stop words with an attribute of the ‘doc’ object called ‘is_stop’. We iterate over all the tokens and apply the ‘is_stop’ method [2].Stop words, such as “the,” “and,” “is,” and “an,” are common words that appear frequently in a language. These terms are frequently irrelevant to the analysis and can be removed to reduce the noise in the data. The NLTK library includes a list of English stop words for this purpose.
<pre>
for token in doc:
 if token.is_stop == True:
  print(token)

</pre>

output:

<pre>
is 
a 
to 
with 

</pre>

These are the most common words which do not add much value to the meaning of the document.[1]

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/80b3b54c-efca-4e01-b838-4adb119443bb_587x375.jpg"></a>
</p>

<pre>
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  stopwords = set(stopwords.words('english'))
</pre>

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/2062e64d-6d4d-4afe-9187-97fac59bf95b_700x216.jpg"></a>
</p>

Let’s take a look at how you can do this. Let’s process this tweet. First, I remove all the words that don’t add significant meaning to the tweets, aka stop words and punctuation marks. In practice, you would have to compare your tweet against two lists. One with stop words in English and another with punctuation. These lists are usually much larger, but for the purpose of this example, they will do just fine. Every word from the tweet that also appears on the list of stop words should be eliminated. So you’d have to eliminate the word and, the word are, the word a, and the word at. The tweet without stop words looks like this.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/188b1d07-caf1-49e4-b752-e785224ba9f4_700x238.jpg"></a>
</p>

Note that the overall meaning of the sentence could be inferred without any effort. Now, let’s eliminate every punctuation mark. In this example, there are only exclamation points. The tweet without stop words and punctuation looks like this.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/a97af959-0dde-4b44-882c-d71112da819b_700x228.jpg"></a>
</p>

However, note that in some contexts you won’t have to eliminate punctuation. So you should think carefully about whether punctuation adds important information to your specific NLP task or not. Tweets and other types of texts often have handles and URLs, but these don’t add any value for the task of sentiment analysis. Let’s eliminate these two handles and this URL. At the end of this process, the resulting tweets contains all the important information related to its sentiment.

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/1df892f7-a71a-4129-8ef2-ced2dbf937f0_657x124.jpg"></a>
</p>
Tuning the GREAT AI model is clearly a positive tweet and a sufficiently good model should be able to classify it.

## 5-Lemmatization

Lemmatization is better than stemming and informative to find beyond the word to its stem also determine part of speech around a word. That’s why spacy has lemmatization, not stemming. So we will do lemmatization with spacy.Lemmatization is another important preprocessing step for NLP pipelines. It helps to remove different versions of a single word to reduce redundancy of same-meaning words as it converts the words to their root lemmas. For example, it will convert ‘is’ -> ‘be’, ‘eating’ -> ‘eat’, and ‘N.Y.’ -> ‘n.y.’. With spaCy, the words can be easily converted to their lemmas using a ‘.lemma_’ attribute of the ‘doc’ object.[2]. We iterate over all the tokens and apply the ‘.lemma_’ method.[2].Lemmatization is the process of reducing a word to its base or root form, called a lemma. Stemming is a similar process, but it often results in words that are not actual words.

For example, the words “walked”, “walking”, and “walk” would all be lemmatized to the word “walk”. This is because they all have the same lemma, which is the dictionary form of the word.

Lemmatization can be done using a variety of tools and techniques. Some popular lemmatizers include the Porter stemmer, the Snowball stemmer, and the WordNet lemmatizer.

Lemmatization is a similar process to stemming, but it reduces words to their base form by using a dictionary or knowledge of the language. This can result in more accurate base forms than stemming [6].

<pre>
for token in doc:
 print(token.lemma_)
output: 

kdnugget 
be 
a 
wonderful 
website 
to 
learn 
machine 
learning 
with 
python
  </pre>
Another text preprocessing technique using which we reduce the words down to their root forms.[1]

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/e4891e81-bdb0-40dc-8837-821a47d7a631_321x168.jpg"></a>
</p>
Code
<pre>
from nltk.stem import WordNetLemmatizer
tokenized = ["booking", "studying", "jumping"]
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(token) for token in tokenized]

Output —
['book','study','jump]
</pre>

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/ee.jpg"></a>
</p>

<p align="center">
<img src="https://github.com/dr-mushtaq/natural-language-processing-projects-python/blob/main/%F0%9F%93%9AChapter%202%20Sentiment%20Analysis%20(Text%20Classification)/FFF.png"></a>
</p>

## A basic example demonstrating how a lemmatizer works

In the following example, we are taking the PoS tag as “verb,” and when we apply the lemmatization rules, it gives us dictionary words instead of truncating the original word:[4]


5-Stemming
What is stemming: Stemming is a process in which words are reduced to their root meaning.It’s a technique to get to the root form of a word by removing the prefix and suffix of a word.[1].We use Stemming to normalize words. In English and many other languages, a single word can take multiple forms depending upon the context used. For instance, the verb “study” can take many forms like “studies,” “studying,” “studied,” and others, depending on its context. When we tokenize words, an interpreter considers these input words as different words even though their underlying meaning is the same. Moreover, as we know that NLP is about analyzing the meaning of content, to resolve this problem, we use stemming [4].

Stemming normalizes the word by truncating the word to its stem word. For example, the words “studies,” “studied,” “studying” will be reduced to “studi,” making all these word forms to refer to only one token. Notice that stemming may not give us a dictionary, grammatical word for a particular set of words [4].In Natural Language Processing (NLP), “steaming” refers to the process of reducing a word to its base or root form. This is often done to group together different forms of a word so they can be analyzed together as a single item [6].

Stemming is the process of reducing words to their base or stem form, by removing any prefixes or suffixes. This is a common technique for reducing the dimensionality of the data, as it groups similar words together.



from nltk.stem import PorterStemmer
tokenized = ["booking", "studying", "jumping"]
stemmer = PorterStemmer()
s = [stemmer.stem(token) for token in tokenized]
['book','studi','jump]
Now that the tweet from the example has only the necessary information, I will perform stemming for every word.

Stemming in NLP is simply transforming any word to its base stem, which you could define as the set of characters that are used to construct the word and its derivatives. Let’s take the first
word from the example. Its stem is tun, because adding the letter e, it forms the word tune. Adding the suffix ed, forms the word tuned, and adding the suffix ing, it forms the word tuning. After you perform
stemming on your corpus, the word tune, tuned, and tuning will be reduced to the stem tun. So your vocabulary would be significantly reduced when you perform this process for every word in the corpus.



To reduce your vocabulary even further without losing valuable information, you’d have to lowercase every one of your words. So the word GREAT, Great and great would be treated as the same exact word. This is the final preprocess tweet as a list of words. Now that you’re familiar with stemming and stop words, you know the basics of texts processing.



Types of stemmer

Porter Stemmer

Snowball Stemmer

Porter stemmer was developed in 1980. It is used for the reduction of a word to its stem or root word.one thing is noticed that the porter stemmer is not giving many good results. So, that’s why the Snowball stemmer is used for a more improved method.

6- Bag of Words
Def: it is a commonly used model that allows you to count all words in a piece of text. Basically, it creates an occurrence matrix for the sentence or document, disregarding grammar and word order. These word frequencies or occurrences are then used as features for training a classifier.

Def: Bag of Words is a text-processing methodology that extracts features from textual data. It uses a pre-defined dictionary of words to measure the presence of known words in your data and doesn’t consider the order of word appearance.

Def: We call vectorization the general process of turning a collection of text documents into numerical feature vectors. This specific strategy (tokenization, counting and normalization) is called the Bag of Words or “Bag of n-grams” representation. Documents are described by word occurrences while completely ignoring the relative position information of the words in the document.

We will use CountVectorizer to convert text into a matrix of token count.

Concepts

The algorithm uses this dictionary to loop through all the documents in the data and can use a simple scoring method to create the vectors. For example, it can mark the presence of a word in a vocabulary as 1 or 0 if absent. Additional scoring methods include looking at the frequency of each word appearing in the document.

Here is an example of a bag-of-words representation of the sentence “John likes to watch movies. Mary likes movies too”:

{'john': 1, 'likes': 2, 'movies': 2, 'mary': 1}
This representation tells us that the words “john”, “likes”, “movies”, and “mary” appear in the sentence, and that the word “likes” appears twice. It does not tell us anything about the order of the words in the sentence, or about the grammatical relationships between the words.

The bag-of-words model is a simple and efficient way to represent text for use in machine learning algorithms. It is often used in tasks such as document classification, sentiment analysis, and topic modeling.

Advantages

It is simple to understand and implement like OneHotEncoding.

We have a fixed length encoding for any sequence of arbitrary length.

Documents with same words/vocabulary will have similar representation. So if two documents have a similar vocabulary, they’ll be closer to each other in the vector space and vice versa.

Source
1-Day 2: 30 Days of Natural Language Processing Series with Projects

2-Getting Started with spaCy for NLP
3-Fully Explained Regular Expression with Python (Unread)

4-Natural Language Processing (NLP) with Python — Tutorial( Unread)

5-Python for Natural Language Processing: A Beginner’s Guide

6-Every Beginner NLP Engineer must know these Techniques (Unread)

7-A Guide to Text Preprocessing Techniques in NLP

8- Natural Language Processing with Classification and Vector Spaces

9–6-How to Convert Text Into Vectors

10-Text Preprocessing For NLP Part — 1

11-Comprehensive Hands on Guide to Twitter Sentiment Analysis with dataset and code

12- Natural Language Processing with Classification and Vector Spaces

13-Use Sentiment Analysis With Python to Classify Movie Reviews

14- How to Create a Custom Tokenizer for Non-English Languages with Hugging Face Transformers(unread)




























