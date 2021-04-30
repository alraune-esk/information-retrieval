from collections import defaultdict
import os
import pickle


def get_zero():
    return 0


def get_empty_postings():
    return defaultdict(get_zero)


class SparseMatrix:
    """
    Used to represent a frequency count matrix.
    token_to_doc_counts maps a token (row) to a dict which maps a doc (column) to a count.
    doc_to_token_counts maps a doc (column) to a dict which maps a token (row) to a count.
    Both of these dicts contain the same data, they just allow for different accessing methods (rows vs columns).
    """
    def __init__(self):
        self.token_to_doc_counts = defaultdict(get_empty_postings)
        self.doc_to_token_counts = defaultdict(get_empty_postings)
        self.num_docs = 0


class InvertedIndex:
    """
    Handles reading raw text files into inverted index form,
    as well as running queries over the created inverted index.
    """
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.similarity_measure = None
        self.postings = SparseMatrix()

    def index_directory(self, directory, use_stored_index=True):
        """
        Grab every file inside directory and add them to the index.
        After indexing is finished, the created SparseMatrix is written to a .pkl file.
        If use_stored_index=True and a .pkl file exists for this directory then the inverted index
        will be read from the file instead of running the indexing process again.
        """
        store_file = f'{directory}_inverted_index.pkl'
        if use_stored_index and os.path.exists(store_file):
            print(f'Loading index from {store_file}.')
            with open(store_file, 'rb') as f:
                self.postings = pickle.load(f)
        else:
            for path, subdirs, files in os.walk(directory):
                print(f'Indexing dir: {path}')
                for file in files:
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as fr:
                        self.index_document(file, fr.read())
            with open(store_file, 'wb') as f:
                pickle.dump(self.postings, f)

    def index_document(self, doc, text):
        tokens = self.preprocessor(text)
        for token in tokens:
            self.postings.token_to_doc_counts[token][doc] += 1
            self.postings.doc_to_token_counts[doc][token] += 1
        self.postings.num_docs += 1

    def run_query(self, query, max_results_returned=10):
        """
        :param query: string of text to be queried for.
        :param max_results_returned: the maximum number of documents to return.
        :return: list of pairs of (document, similarity), for the max_results_returned most similar documents.
        """
        query_tokens = self.preprocessor(query)
        query_vector = defaultdict(lambda: 0)
        for token in query_tokens:
            query_vector[token] += 1
        sim_scores = self.similarity_measure(query_vector)
        sorted_sim_scores = sorted(sim_scores, key=sim_scores.get, reverse=True)
        # return a pair of doc and similarity score in descending order
        sorted_sim_scores = [(doc, sim_scores[doc]) for doc in sorted_sim_scores]
        try:
            return sorted_sim_scores[:max_results_returned]
        except:
            print("Error with the input number of results returned")

    def set_similarity(self, sim):
        self.similarity_measure = sim(self.postings)
