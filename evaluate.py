import os
from trectools import TrecQrel, procedures

qrels_file = os.path.join('gov', 'qrels', 'gov.qrels')
qrels = TrecQrel(qrels_file)

path_to_runs = 'runs'
runs = procedures.list_of_runs_from_path(path_to_runs, '*.runs')
results = procedures.evaluate_runs(runs, qrels, per_query=False)

metrics = ['map', 'Rprec', 'recip_rank', 'P_5', 'P_10', 'P_15']
for metric in metrics:
    print(f'{metric}: {procedures.extract_metric_from_results(results, metric)[0][1]}')
