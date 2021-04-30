from abc import abstractmethod
from collections import defaultdict
from math import log, sqrt


class CosineSimilarity:
    """
    This class calculates a similarity score between a given query and all documents in an inverted index.
    """
    def __init__(self, postings):
        self.postings = postings
        self.doc_to_norm = dict()
        self.set_document_norms()

    def __call__(self, query):
        doc_to_score = defaultdict(lambda: 0)
        self.get_scores(doc_to_score, query)
        return doc_to_score

    @abstractmethod
    def set_document_norms(self):
        """
        Set self.doc_to_norm to contain the norms of every document.
        """
        pass

    @abstractmethod
    def get_scores(self, doc_to_score, query):
        """
        For each document add an entry to doc_to_score with this document's similarity to query.
        """
        pass


class BM25:
    """
        This class calculates the BM25 similarity score between a given query and all documents in an inverted index.
    """
    def __init__(self, postings):
        self.postings = postings
        self.doc_tokens_count = dict()
        self.avg_doc_length = 0
        self.set_doc_token_counts()

    def __call__(self, query):
        doc_to_score = defaultdict(lambda: 0)
        self.get_scores(doc_to_score, query)
        return doc_to_score

    @abstractmethod
    def set_doc_token_counts(self):
        """
        Set self.doc_tokens_count to contain the number of tokens in every document.
        """
        pass

    @abstractmethod
    def get_scores(self, doc_to_score, query):
        """
        For each document add an entry to doc_to_score with this document's similarity to query.
        """
        pass


class TF_Similarity(CosineSimilarity):

    def set_document_norms(self):
        for doc, token_counts in self.postings.doc_to_token_counts.items():
            self.doc_to_norm[doc] = sqrt(sum([tf ** 2 for tf in token_counts.values()]))

    def get_scores(self, doc_to_score, query):
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                doc_to_score[doc] += query_term_frequency * document_term_frequency / self.doc_to_norm[doc]


class TFIDF_Similarity(CosineSimilarity):
    # set the norms for tf idf similarity
    def set_document_norms(self):
        for term, doc_counts in self.postings.token_to_doc_counts.items():
            for doc, counts in doc_counts.items():
                # if the term does not exist in the document, then the idf norm is 0
                if doc in self.doc_to_norm:
                    self.doc_to_norm[doc] += (counts * log(self.postings.num_docs/len(doc_counts))) ** 2
                else:
                    self.doc_to_norm[doc] = 0
        for doc, norm in self.doc_to_norm.items():
            self.doc_to_norm[doc] = sqrt(norm)

    # set scores for each doc using tf idf similarity calculation
    def get_scores(self, doc_to_score, query):
        # calculate the tf idf calculation accumulating for each token and for each document
        for token, query_term_frequency in query.items():
            for doc, document_term_frequency in self.postings.token_to_doc_counts[token].items():
                idf = log(self.postings.num_docs/len(self.postings.token_to_doc_counts[token]))
                if self.doc_to_norm[doc] == 0:
                    doc_to_score[doc] += 0
                else:
                    doc_to_score[doc] += query_term_frequency * idf ** 2 * document_term_frequency / \
                                         self.doc_to_norm[doc]


class BM25_Similarity(BM25):
    """
    Set the number of tokens for each document and use this alongside the query and postings
    to calculate a BM25 similarity score for each document
    """
    def set_doc_token_counts(self):
        total_count = 0
        # count the number of tokens in each document and in total across all documents
        for doc, token_dict in self.postings.doc_to_token_counts.items():
            doc_count = 0
            for token, count in token_dict.items():
                doc_count += count
            self.doc_tokens_count[doc] = doc_count
            total_count += doc_count
        self.avg_doc_length = total_count/len(self.postings.doc_to_token_counts)

    def get_scores(self, doc_to_score, query):
        k1 = 2
        b = 0.75
        # for each token accumulate for each document the BM25 similarity score
        for token, query_term_frequency in query.items():
            for doc, doc_term_frequency in self.postings.token_to_doc_counts[token].items():
                idf = log(self.postings.num_docs/len(self.postings.token_to_doc_counts[token]))
                doc_to_score[doc] += idf * doc_term_frequency * (k1 + 1) / \
                        (doc_term_frequency + k1 * (1 - b + b * self.doc_tokens_count[doc] / self.avg_doc_length))