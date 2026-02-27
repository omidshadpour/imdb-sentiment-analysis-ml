import re
import string
import nltk
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer


class TextPreprocessor:
    def __init__(self):
        nltk.download('punkt' , quiet=True)
        nltk.download('punkt_tab' , quiet=True)
        nltk.download('wordnet' , quiet=True)
        nltk.download('omw-1.4' , quiet=True)

        self.lemmatizer = WordNetLemmatizer()


    def clean_text(self, text: str):
        text= text.lower()
        text = re.sub(r"http\S+|www\S+" , "" , text)
        text = re.sub(r"<.*?>" , "" , text)
        text = text.translate(str.maketrans("" , "" , string.punctuation))
        text = re.sub(r"\s+" , " " , text).strip()

        return text

    def lemmatize_text(self, text: str):
        tokens = word_tokenize((text))
        lemmatized = [self.lemmatizer.lemmatize(token) for token in tokens]
        return " ".join(lemmatized)

    def preprocess_series(self, text_series):
        return text_series.apply(lambda x: self.lemmatize_text(self.clean_text(x)))

