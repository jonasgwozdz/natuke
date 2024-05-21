import pandas as pd
from ast import literal_eval
import numpy as np

def hits_at(k, true, list_pred):
    hits = []
    missed_entries = []

    for index_t, t in enumerate(true):
        hit = False
        # get the list of predicteds that's on the second argument
        for index_lp, lp in enumerate(list_pred[index_t][1]):
            if index_lp >= k:
                break
            if t[1] == lp:
                hits.append(1)
                hit = True
                break
        if not(hit):
            hits.append(0)
            missed_entries.append((index_t, t, list_pred[index_t][1][:k]))
    return np.mean(hits), missed_entries

def mrr(true, list_pred):
    # using the first list pred to get how many there will be
    rrs = []
    missed_entries = []
    for index_t, t in enumerate(true):
        hit = False
        # get the list of predicteds that's on the second argument
        for index_lp, lp in enumerate(list_pred[index_t][1]):
            if t[1] == lp:
                rrs.append(1/(index_lp + 1))
                break
            if not hit:
                missed_entries.append((index_t, t, list_pred[index_t][1]))

    return np.mean(rrs), missed_entries


path = 'natuke_data/'
file_name = "llm_results"
splits = [0.8]
#edge_groups = ['doi_name', 'doi_bioActivity', 'doi_collectionSpecie', 'doi_collectionSite', 'doi_collectionType']
edge_group = 'doi_collectionSite'
#algorithms = ['bert', 'deep_walk', 'node2vec', 'metapath2vec', 'regularization']
algorithms = ['gpt4']
k_at = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
dynamic_stages = ['1st', '2nd', '3rd', '4th']

# hits@k
hitsatk_df = {'k': [], 'algorithm': [], 'edge_group': [], 'split': [], 'dynamic_stage': [], 'value': []}
missed_hits = []

for algorithm in algorithms:
    for k in k_at:
        for split in splits:
            for iteration in range(1):
                for dynamic_stage in dynamic_stages:
                    restored_df = pd.read_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, dynamic_stage))
                    restored_df['true'] = restored_df['true'].apply(literal_eval)
                    restored_df['restored'] = restored_df['restored'].apply(literal_eval)
                    mean_hits, missed = hits_at(k, restored_df.true.to_list(), restored_df.restored.to_list())
                    hitsatk_df['k'].append(k)
                    hitsatk_df['algorithm'].append(algorithm)
                    hitsatk_df['split'].append(split)
                    hitsatk_df['edge_group'].append(edge_group)
                    hitsatk_df['dynamic_stage'].append(dynamic_stage)
                    hitsatk_df['value'].append(mean_hits)
                    missed_hits.extend(missed)

hitsatk_df = pd.DataFrame(hitsatk_df)
hitsatk_df.to_csv('{}metric_results/full_dynamic_hits@k_{}_{}.csv'.format(path, edge_group, file_name), index=False)
hitsatk_df_mean = hitsatk_df.groupby(by=['k', 'algorithm', 'split', 'edge_group', 'dynamic_stage'], as_index=False).mean()
hitsatk_df_std = hitsatk_df.groupby(by=['k', 'algorithm', 'split', 'edge_group', 'dynamic_stage'], as_index=False).std()
hitsatk_df_mean['std'] = hitsatk_df_std['value']
print(hitsatk_df_mean)

# Save missed hits entries for debugging
missed_hits_df = pd.DataFrame(missed_hits, columns=['index', 'true', 'predictions'])
missed_hits_df.to_csv('{}metric_results/missed_hits@k_{}_{}.csv'.format(path, edge_group, file_name), index=False)

# mrr
mrr_df = {'algorithm': [], 'edge_group': [], 'split': [], 'dynamic_stage': [], 'value': []}
missed_mrr = []

for algorithm in algorithms:
    for split in splits:
        for iteration in range(1):
            for dynamic_stage in dynamic_stages:
                restored_df = pd.read_csv("{}results/{}_{}_{}_{}_{}_{}.csv".format(path, file_name, algorithm, split, edge_group, iteration, dynamic_stage))
                restored_df['true'] = restored_df['true'].apply(literal_eval)
                restored_df['restored'] = restored_df['restored'].apply(literal_eval)
                mean_mrr, missed = mrr(restored_df.true.to_list(), restored_df.restored.to_list())
                mrr_df['algorithm'].append(algorithm)
                mrr_df['split'].append(split)
                mrr_df['edge_group'].append(edge_group)
                mrr_df['dynamic_stage'].append(dynamic_stage)
                mrr_df['value'].append(mean_mrr)
                missed_mrr.extend(missed)

mrr_df = pd.DataFrame(mrr_df)
mrr_df.to_csv('{}metric_results/full_dynamic_mrr_{}_{}.csv'.format(path, edge_group, file_name), index=False)
mrr_df_mean = mrr_df.groupby(by=['algorithm', 'edge_group', 'split', 'dynamic_stage'], as_index=False).mean()
mrr_df_std = mrr_df.groupby(by=['algorithm', 'edge_group', 'split', 'dynamic_stage'], as_index=False).std()
mrr_df_mean['std'] = mrr_df_std['value']
print(mrr_df_mean)

# Save missed MRR entries for debugging
missed_mrr_df = pd.DataFrame(missed_mrr, columns=['index', 'true', 'predictions'])
missed_mrr_df.to_csv('{}metric_results/missed_mrr_{}_{}.csv'.format(path, edge_group, file_name), index=False)

# saving files
hitsatk_df_mean.to_csv('{}metric_results/dynamic_hits@k_{}_{}.csv'.format(path, edge_group, file_name), index=False)
mrr_df_mean.to_csv('{}metric_results/dynamic_mrr_{}_{}.csv'.format(path, edge_group, file_name), index=False)