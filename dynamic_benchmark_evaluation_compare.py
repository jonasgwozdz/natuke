import pandas as pd
from ast import literal_eval
import numpy as np

def hits_at(k, true, list_pred):
    hits = []
    missed_entries = []

    for index_t, t in enumerate(true):
        hit = False
        for index_lp, lp in enumerate(list_pred[index_t][1]):
            if index_lp >= k:
                break
            if t[1] == lp:
                hits.append(1)
                hit = True
                break
        if not hit:
            hits.append(0)
            missed_entries.append((index_t, t, list_pred[index_t][1][:k]))
    return np.mean(hits), missed_entries

def mrr(true, list_pred):
    rrs = []
    missed_entries = []
    for index_t, t in enumerate(true):
        hit = False
        for index_lp, lp in enumerate(list_pred[index_t][1]):
            if t[1] == lp:
                rrs.append(1/(index_lp + 1))
                hit = True
                break
        if not hit:
            missed_entries.append((index_t, t, list_pred[index_t][1]))
    return np.mean(rrs), missed_entries

def evaluate_results(path, file_name, algorithms, splits, edge_group, k_at, dynamic_stages, result_label):
    hitsatk_df = {'k': [], 'algorithm': [], 'edge_group': [], 'split': [], 'dynamic_stage': [], 'value': [], 'result': []}
    missed_hits = []
    mrr_df = {'algorithm': [], 'edge_group': [], 'split': [], 'dynamic_stage': [], 'value': [], 'result': []}
    missed_mrr = []

    for algorithm in algorithms:
        for k in k_at:
            for split in splits:
                for iteration in range(10):
                    for dynamic_stage in dynamic_stages:
                        restored_df = pd.read_csv(f"{path}{file_name}_{algorithm}_{split}_{edge_group}_{iteration}_{dynamic_stage}.csv")
                        restored_df['true'] = restored_df['true'].apply(literal_eval)
                        restored_df['restored'] = restored_df['restored'].apply(literal_eval)
                        
                        mean_hits, missed_hits_entries = hits_at(k, restored_df.true.to_list(), restored_df.restored.to_list())
                        hitsatk_df['k'].append(k)
                        hitsatk_df['algorithm'].append(algorithm)
                        hitsatk_df['split'].append(split)
                        hitsatk_df['edge_group'].append(edge_group)
                        hitsatk_df['dynamic_stage'].append(dynamic_stage)
                        hitsatk_df['value'].append(mean_hits)
                        hitsatk_df['result'].append(result_label)
                        missed_hits.extend([(entry[0], entry[1], entry[2], result_label) for entry in missed_hits_entries])
                        
                        mean_mrr, missed_mrr_entries = mrr(restored_df.true.to_list(), restored_df.restored.to_list())
                        mrr_df['algorithm'].append(algorithm)
                        mrr_df['split'].append(split)
                        mrr_df['edge_group'].append(edge_group)
                        mrr_df['dynamic_stage'].append(dynamic_stage)
                        mrr_df['value'].append(mean_mrr)
                        mrr_df['result'].append(result_label)
                        missed_mrr.extend([(entry[0], entry[1], entry[2], result_label) for entry in missed_mrr_entries])

    return pd.DataFrame(hitsatk_df), pd.DataFrame(mrr_df), missed_hits, missed_mrr

path1 = 'natuke_data/results1/'
path2 = 'natuke_data/results2/'
file_name = "llm_results"
splits = [0.8]
edge_group = 'doi_collectionSite'
algorithms = ['gpt4']
k_at = [20]
dynamic_stages = ['1st', '2nd', '3rd', '4th']

hitsatk_df1, mrr_df1, missed_hits1, missed_mrr1 = evaluate_results(path1, file_name, algorithms, splits, edge_group, k_at, dynamic_stages, 'Result1')
hitsatk_df2, mrr_df2, missed_hits2, missed_mrr2 = evaluate_results(path2, file_name, algorithms, splits, edge_group, k_at, dynamic_stages, 'Result2')

hitsatk_df_combined = pd.concat([hitsatk_df1, hitsatk_df2])
mrr_df_combined = pd.concat([mrr_df1, mrr_df2])

hitsatk_df_combined.to_csv('natuke_data/metric_results/combined_dynamic_hits@k_{}_{}.csv'.format(edge_group, file_name), index=False)
mrr_df_combined.to_csv('natuke_data/metric_results/combined_dynamic_mrr_{}_{}.csv'.format(edge_group, file_name), index=False)

missed_hits_combined = missed_hits1 + missed_hits2
missed_hits_df = pd.DataFrame(missed_hits_combined, columns=['index', 'true', 'predictions', 'result'])
missed_hits_df.to_csv('natuke_data/metric_results/combined_missed_hits@k_{}_{}.csv'.format(edge_group, file_name), index=False)

missed_mrr_combined = missed_mrr1 + missed_mrr2
missed_mrr_df = pd.DataFrame(missed_mrr_combined, columns=['index', 'true', 'predictions', 'result'])
missed_mrr_df.to_csv('natuke_data/metric_results/combined_missed_mrr_{}_{}.csv'.format(edge_group, file_name), index=False)

# Create a DataFrame to store the comparison of mistakes between results1 and results2
comparison_mistakes = []

# Iterate through missed_hits2 to find corresponding entries in missed_hits1
for entry2 in missed_hits2:
    index2, true2, predictions2, result2 = entry2
    corresponding_entry1 = next((entry1 for entry1 in missed_hits1 if entry1[0] == index2), None)
    if corresponding_entry1:
        index1, true1, predictions1, result1 = corresponding_entry1
        comparison_mistakes.append([index2, true2, predictions1, predictions2])

# Save the comparison of mistakes to a CSV file
comparison_mistakes_df = pd.DataFrame(comparison_mistakes, columns=['index', 'true', 'prediction1', 'prediction2'])
comparison_mistakes_df.to_csv('natuke_data/metric_results/comparison_mistakes_{}_{}.csv'.format(edge_group, file_name), index=False)

hitsatk_df_mean = hitsatk_df_combined.groupby(by=['k', 'algorithm', 'split', 'edge_group', 'dynamic_stage', 'result'], as_index=False).mean()
hitsatk_df_std = hitsatk_df_combined.groupby(by=['k', 'algorithm', 'split', 'edge_group', 'dynamic_stage', 'result'], as_index=False).std()
hitsatk_df_mean['std'] = hitsatk_df_std['value']
print(hitsatk_df_mean)

mrr_df_mean = mrr_df_combined.groupby(by=['algorithm', 'edge_group', 'split', 'dynamic_stage', 'result'], as_index=False).mean()
mrr_df_std = mrr_df_combined.groupby(by=['algorithm', 'edge_group', 'split', 'dynamic_stage', 'result'], as_index=False).std()
mrr_df_mean['std'] = mrr_df_std['value']
print(mrr_df_mean)

hitsatk_df_mean.to_csv('natuke_data/metric_results/mean_dynamic_hits@k_{}_{}.csv'.format(edge_group, file_name), index=False)
mrr_df_mean.to_csv('natuke_data/metric_results/mean_dynamic_mrr_{}_{}.csv'.format(edge_group, file_name), index=False)
