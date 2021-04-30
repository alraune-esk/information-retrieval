import nltk
from functools import lru_cache
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import re

class Preprocessor:
    def __init__(self):
        # Stemming is the most time-consuming part of the indexing process, we attach a lru_cache to the stemmer
        # which will store upto 100000 stemmed forms and reuse them when possible instead of applying the
        # stemming algorithm.
        self.stem = lru_cache(maxsize=10000)(nltk.PorterStemmer().stem)
        self.tokenize = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

        # Obtain a dictionary containing the (abbreviation : expanded abbreviation)
        # of common abbreviations in the corpus
        # uncomment below to use
        # abbreviation_dic = {}
        # with open('abbreviations.txt', 'r') as reader:
        #     for line in reader.readlines():
        #         line_split = line.split()
        #         abbrev = line_split[0]
        #         abbrev_expanded = " ".join(line_split[1:])
        #         abbreviation_dic[abbrev] = abbrev_expanded
        # self.abbreviations = abbreviation_dic

    def __call__(self, text):

        # remove email addresses
        text = re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", ' ', str(text))

        # apply the Regexp tokenizer
        tokens = nltk.tokenize.RegexpTokenizer(r'\w+').tokenize(text)

        # Abbreviation expanding code, add to the list of tokens matched expanded abbreviations
        # uncomment below to use
        # for token in tokens:
        #     if token in self.abbreviations.keys():
        #         abbrev_expanded = self.abbreviations[token]
        #         for term in abbrev_expanded.split():
        #             tokens.append(term)

        # Lower case normalisation
        tokens = [token.lower() for token in tokens]

        # Stop word removal
        tokens = [token for token in tokens if token not in self.stop_words]

        # Porter Stemming
        tokens = [self.stem(token) for token in tokens]

        # Pos tagged wordnet lemmatization
        # Note the tagged lemmatization takes 10minutes + to run
        tagged_tokens = pos_tag(tokens)
        tokens = []
        for token, tag in tagged_tokens:
            postag = tag[0].lower()
            if postag in ['r', 'a', 'v', 'n']:
                lemma = self.lemmatizer.lemmatize(token, postag)
            else:
                lemma = token
            tokens.append(lemma)

        return tokens
