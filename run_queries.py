import argparse
import os

from inverted_index import InvertedIndex
from preprocessor import Preprocessor
from similarity_measures import TF_Similarity, TFIDF_Similarity, BM25_Similarity

parser = argparse.ArgumentParser(description='Run all queries on the inverted index.')
parser.add_argument('--new', default=True, help='If True then build a new index from scratch. If False then attempt to'
                                                ' reuse existing index')
parser.add_argument('--sim', default='BM25', help='The type of similarity to use. Should be "TF" or "TFIDF" or "BM25"')
args = parser.parse_args()

index = InvertedIndex(Preprocessor())
index.index_directory(os.path.join('gov', 'documents'), use_stored_index=(not args.new))

sim_name_to_class = {'TF': TF_Similarity,
                     'TFIDF': TFIDF_Similarity,
                     'BM25': BM25_Similarity}

sim = sim_name_to_class[args.sim]
index.set_similarity(sim)
print(f'Setting similarity to {sim.__name__}')

print()
print('Index ready.')


topics_file = os.path.join('gov', 'topics', 'gov.topics')
runs_file = os.path.join('runs', 'retrieved.runs')

# Read the topics file and write to the runs file
topics = open(topics_file, 'r')
topic_lines = topics.readlines()
trec = open(runs_file, "w")

# for each topic (query) retrieve the query id as well as the
# sorted document and scores
for line in topic_lines:
    query = line.split()
    query_id = query[0]
    # disregard the query id as it is not part of the query
    query = " ".join(query[1:])
    sorted_results = index.run_query(query)

# write the results of the query in trec eval format
    for rank, doc_and_scores in enumerate(sorted_results):
        doc = str(doc_and_scores[0])
        score = str(doc_and_scores[1])
        trec.write(str(query_id) + " Q0 " + doc + " " + str(rank) + " " + score + " MY_IR_SYSTEM\n")
topics.close()
trec.close()
