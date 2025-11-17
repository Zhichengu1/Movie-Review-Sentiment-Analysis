📚 1. Required Python Packages (Explained)
pandas

Used to load and manipulate datasets (tables).
Used in:

loading CSV files

selecting columns

storing reviews

👉 Docs: https://pandas.pydata.org/docs/

numpy

Used for numerical operations.
Mostly supports ML/NN models behind the scenes.

👉 Docs: https://numpy.org/doc/

re (Regular Expressions)

Used to remove:

HTML tags

special characters

numbers

Example:
<br /> → removed
!!! → removed

👉 Docs: https://docs.python.org/3/library/re.html

NLTK (Natural Language Toolkit)

We use it for basic NLP:

NLTK Tool	What It Does
stopwords	Removes words like “the”, “is”, “and”
word_tokenize()	Splits sentence → words
WordNetLemmatizer()	Converts word to base form (“cars” → “car”)

👉 Docs: https://www.nltk.org/

scikit-learn

Used for:

TF-IDF (turn text into numbers)

training ML models

train/test split

evaluation metrics

👉 Docs: https://scikit-learn.org/stable/

PyTorch

Used for building and training the LSTM model.

👉 Docs: https://pytorch.org/docs/