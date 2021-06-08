from load_website_data import load_icl
import numpy as np
try:
    from scipy.io import savemat
except ImportError:
    pass
from crowd_labeling import CLLDA, concurrent_cllda, combine_cllda
from crowd_labeling.MV import MV
import cPickle as pkl
import json
from os.path import join, isfile, isdir
import argparse
from copy import deepcopy
import sys

# parse input arguments
print('parsing arguments')
parser = argparse.ArgumentParser(description='Run or update CL estimates.')
parser.add_argument('database', type=str, help='Absolute reference to the sqlite database file.')
parser.add_argument('save', type=str, help='Directory in which to save results.')
parser.add_argument('-classifications', type=str, help='Directory in which to save results for website viewing.',
                    default=None)
args = parser.parse_args()
database = args.database
path = args.save
classifications_path = args.classifications
assert isfile(database), 'database path does not exist'
assert isdir(path), 'save path does not exist'
assert isdir(classifications_path), 'classifications path does not exist'

# load sqlite data
print('loading database')
icl_votes = load_icl(database)
votes = icl_votes['votes']
workers = icl_votes['workers']
instances = icl_votes['instances']
instance_set_numbers = icl_votes['instance_set_numbers']
instance_ic_numbers = icl_votes['instance_ic_numbers']
vote_ids = np.array(['Brain', 'Muscle', 'Eye', 'Heart', 'Line Noise', 'Chan Noise', 'Other', '?'])
instance_ids = icl_votes['instance_ids']
worker_ids = icl_votes['worker_ids']
T = icl_votes['n_classes']
C = icl_votes['n_responses']
A = icl_votes['n_workers']
is_expert = (icl_votes['is_expert']).astype(bool)  # type = np.ndarray


# append identifier to string
def add_identifier(string, identifier):
    return '_'.join((x for x in (string, identifier) if x is not None))


# run cllda
def run_cllda(save_path, vts, wks, its, vt_ids=None, it_ids=None, wk_ids=None, it_priors=None, wk_priors=None,
              it_set_numbers=None, it_ic_numbers=None, identifier=None):

    if isfile(join(save_path, add_identifier('icl_cllda_models', identifier) + '.pkl')):
        # load for python
        with open(join(save_path, add_identifier('icl_cllda_models', identifier) + '.pkl'), 'rb') as f:
            cls = pkl.load(f)

        # update CLLDA with all transforms
        cls = concurrent_cllda(cls, vts, wks, its, nprocs=4, vote_ids=vt_ids, instance_ids=it_ids,
                               worker_ids=wk_ids, worker_prior=wk_priors, num_epochs=800, burn_in=0)

    else:
        # CLLDA with all transforms
        cls = concurrent_cllda(4, vts, wks, its, nprocs=4, vote_ids=vt_ids, instance_ids=it_ids,
                               worker_ids=wk_ids, worker_prior=wk_priors, instance_prior=it_priors,
                               transform=('none', 'ilr', 'clr', 'alr'), num_epochs=1000, burn_in=200)

    # save individual models for python
    with open(join(save_path, add_identifier('icl_cllda_models', identifier) + '.pkl'), 'wb') as f:
        pkl.dump(cls, f)

    # combine models
    cl = combine_cllda(cls)

    # aggregate data
    return {
        'instance_ids': cl.instance_ids,
        'worker_ids': cl.worker_ids,
        'vote_ids': cl.vote_ids,
        'instance_set_numbers': it_set_numbers.astype(int),
        'instance_ic_numbers': it_ic_numbers.astype(int),
        'transform': cl.transform,
        'labels': cl.labels,
        'labels_cov': cl.labels_cov,
        'worker_mats': cl.worker_mats,
    }


# save results in 3 different formats
def save_results(save_path, data, identifier=None):
    # save combined model for php
    print('saving for php')
    json_data = deepcopy(data)
    for key, val in json_data.iteritems():
        if isinstance(val, np.ndarray):
            json_data[key] = val.tolist()
        elif isinstance(val, list):
            for it, item in enumerate(val):
                if isinstance(item, np.ndarray):
                    val[it] = item.tolist()
            json_data[key] = val
    with open(join(save_path, add_identifier('ICLabels', identifier) + '.json'), 'wb') as f:
        json.dump(json_data, f)

    # save combined model for python
    print('saving for python')
    with open(join(save_path, add_identifier('ICLabels', identifier) + '.pkl'), 'wb') as f:
        pkl.dump(data, f)

    # save combined model for matlab
    if 'savemat' in sys.modules:
        print('saving for matlab')
        for key, val in data.iteritems():
            if not isinstance(val, np.ndarray):
                try:
                    val = np.array(val)
                except ValueError:
                    data[key] = np.empty(len(val), dtype=np.object)
                    for it, item in enumerate(val):
                        data[key][it] = item
                    continue
            if not np.issubdtype(val.dtype, np.number):
                data[key] = val.astype(np.object)
        savemat(join(save_path, add_identifier('ICLabels', identifier) + '.mat'), data)

    # optionally save classifications for website viewing
    if isdir(classifications_path) and all((x in data.keys() for x in ('labels', 'vote_ids',
                                                                       'instance_set_numbers', 'instance_ic_numbers'))):
        path_str = join(classifications_path, add_identifier('website', identifier) + '_icl_')
        with open(path_str + 'index.json', 'w') as f:
            json.dump(zip(json_data['instance_set_numbers'], json_data['instance_ic_numbers']), f)
        with open(path_str + 'classifications.json', 'w') as f:
            try:
                json.dump(json_data['labels'][np.where(np.array(data['transform']) == 'none')[0][0]], f)
            except KeyError:
                json.dump(json_data['labels'], f)
        with open(path_str + 'classes.json', 'w') as f:
            json.dump(json_data['vote_ids'][:-1], f)


# CLLDA settings
n_pseudovotes_e = 100
n_pseudovotes_u = 1
expert_prior = n_pseudovotes_e * (np.hstack((np.eye(T), np.zeros((T, 1))))) + 0.01
user_prior = n_pseudovotes_u * (np.hstack((np.eye(T), np.zeros((T, 1))))) + 0.01
all_priors = np.zeros((A, T, C))
all_priors[is_expert.astype(np.bool), :, :] = np.tile(expert_prior[np.newaxis], [is_expert.sum(), 1, 1])
all_priors[np.logical_not(is_expert), :, :] = np.tile(user_prior[np.newaxis], [np.logical_not(is_expert).sum(), 1, 1])
instance_prior = np.histogram(votes, range(C))[0] / 100. / np.histogram(votes, range(C))[0].sum()

# run and save CLLDA with experts
tag = 'expert'
print('Running CLLDA_' + tag + '...')
out = run_cllda(path, votes, workers, instances, vote_ids, instance_ids, worker_ids, instance_prior,
                all_priors, instance_set_numbers, instance_ic_numbers, tag)
print('Saved individual CLLDA_' + tag + ' models')
print('Saving combined results...')
save_results(path, out, tag)
print('Saved combined results')


# run CLLDA without experts
tag = 'noexpert'
print('Running CLLDA_' + tag + '...')
out = run_cllda(path, votes, workers, instances, vote_ids, instance_ids, worker_ids, instance_prior,
                user_prior, instance_set_numbers, instance_ic_numbers, tag)
print('Saved individual CLLDA_' + tag + ' models')
print('Saving combined results...')
save_results(path, out, tag)
print('Saved combined results')


# run and save with only luca

# remove non-luca votes
worker_ids_lu = worker_ids[0]
luca_ind = np.in1d(workers, (0,))
votes_lu = votes[luca_ind]
workers_lu = workers[luca_ind]
instances_lu = instances[luca_ind]

# remove instances with votes that are unsure
keep_index = np.logical_not(np.in1d(instances_lu, np.unique(instances_lu[votes_lu == 7])))
votes_lu = votes_lu[keep_index]
workers_lu = workers_lu[keep_index]
instances_lu = instances_lu[keep_index]
instance_ids_lu = instance_ids[np.unique(instances_lu)]

# reset instance numbering
instance_set_numbers_lu = np.array(instance_set_numbers)[np.unique(instances_lu)]
instance_ic_numbers_lu = np.array(instance_ic_numbers)[np.unique(instances_lu)]
instances_lu = np.array([{x: y for x, y in zip(np.unique(instances_lu),
                                               np.arange(np.unique(instances_lu).size))}[z]
                         for z in instances_lu])

# run MV
cl = MV(votes_lu, workers_lu, instances_lu)

# save results
save_results(path, {
    'instance_ids': instance_ids_lu,
    'worker_ids': worker_ids_lu,
    'vote_ids': vote_ids,
    'instance_set_numbers': instance_set_numbers_lu,
    'instance_ic_numbers': instance_ic_numbers_lu,
    'labels': cl.labels,
}, 'onlyluca')

