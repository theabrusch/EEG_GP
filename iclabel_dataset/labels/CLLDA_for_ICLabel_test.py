from load_website_data import load_icl_test
import numpy as np
from crowd_labeling.CLLDA import concurrent_cllda, combine_cllda
import cPickle as pkl
from scipy.io import savemat

# load sqlite data
icl_votes = load_icl_test('database.sqlite')
votes_vec = icl_votes['votes']
votes_vec_workers = icl_votes['workers']
votes_vec_instances = icl_votes['instances']
instance_study_numbers = icl_votes['instance_study_numbers']
instance_set_numbers = icl_votes['instance_set_numbers']
instance_ic_numbers = icl_votes['instance_ic_numbers']
T = icl_votes['n_classes']
C = icl_votes['n_responses']
A = icl_votes['n_workers']

# CLLDA settings
all_priors = np.tile(np.maximum(np.hstack((5*np.eye(T), np.zeros((T, 1)))), 0.01), [A, 1, 1])
instance_prior = np.histogram(votes_vec, range(C))[0] / 100. / np.histogram(votes_vec, range(C))[0].sum()

# CLLDA with all transforms
cls = concurrent_cllda(4, votes_vec, votes_vec_workers, votes_vec_instances, nprocs=4,
                       worker_prior=all_priors, instance_prior=instance_prior,
                       transform=('none', 'ilr', 'clr', 'alr'), num_epochs=1000, burn_in=200)

# combine models
cl = combine_cllda(cls)



# CLLDA with all transforms weak
all_priors_weak = np.tile(np.maximum(np.hstack((np.eye(T), np.zeros((T, 1)))), 0.01), [A, 1, 1])
cls_weak = concurrent_cllda(4, votes_vec, votes_vec_workers, votes_vec_instances, nprocs=4,
                       worker_prior=all_priors_weak, instance_prior=instance_prior,
                       transform=('none', 'ilr', 'clr', 'alr'), num_epochs=1000, burn_in=200)
cl_weak = combine_cllda(cls_weak)

# MV and DS and CLLDA
from crowd_labeling import MV
from crowd_labeling import DS
# ignoring "?"
ind = votes_vec != 7
temp_votes_vec = votes_vec[ind]
temp_votes_vec_workers = votes_vec_workers[ind]
temp_votes_vec_instances = votes_vec_instances[ind]
cls_ignore = concurrent_cllda(4, temp_votes_vec, temp_votes_vec_workers, temp_votes_vec_instances, nprocs=4,
                              worker_prior=all_priors, instance_prior=instance_prior,
                              transform=('none', 'ilr', 'clr', 'alr'), num_epochs=1000, burn_in=200)
cl_ignore = combine_cllda(cls_ignore)
_, temp_votes_vec_workers = np.unique(temp_votes_vec_workers, return_inverse=True)
_, temp_votes_vec_instances = np.unique(temp_votes_vec_instances, return_inverse=True)
mv_ignore = MV(temp_votes_vec, temp_votes_vec_workers, temp_votes_vec_instances)
ds_ignore = DS(temp_votes_vec, temp_votes_vec_workers, temp_votes_vec_instances)
# removing labels with "?"
ind = votes_vec == 7
to_remove = np.stack((votes_vec_workers[ind], votes_vec_instances[ind])).T
ind = np.ones_like(votes_vec, dtype=bool)
for it, vote in enumerate(np.stack((votes_vec_workers, votes_vec_instances)).T):
    if (vote == to_remove).all(1).any():
        ind[it] = False
temp_votes_vec = votes_vec[ind]
temp_votes_vec_workers = votes_vec_workers[ind]
temp_votes_vec_instances = votes_vec_instances[ind]
_, temp_votes_vec_workers = np.unique(temp_votes_vec_workers, return_inverse=True)
_, temp_votes_vec_instances = np.unique(temp_votes_vec_instances, return_inverse=True)
mv_remove = MV(temp_votes_vec, temp_votes_vec_workers, temp_votes_vec_instances)
ds_remove = DS(temp_votes_vec, temp_votes_vec_workers, temp_votes_vec_instances)
cls_remove = concurrent_cllda(4, temp_votes_vec, temp_votes_vec_workers, temp_votes_vec_instances, nprocs=4,
                              worker_prior=all_priors, instance_prior=instance_prior,
                              transform=('none', 'ilr', 'clr', 'alr'), num_epochs=1000, burn_in=200)
cl_remove = combine_cllda(cls_remove)



# results to save
save = dict()
save['instance_labels'] = cl.labels[0]
save['instance_labels_ilr'] = cl.labels[1]
save['instance_labels_clr'] = cl.labels[2]
save['instance_labels_alr'] = cl.labels[3]
save['instance_label_cov'] = cl.labels_cov[0]
save['instance_label_cov_ilr'] = cl.labels_cov[1]
save['instance_label_cov_clr'] = cl.labels_cov[2]
save['instance_label_cov_alr'] = cl.labels_cov[3]
save['instance_id'] = cl.instance_ids
save['instance_number'] = votes_vec_instances
save['instance_study_numbers'] = instance_study_numbers
save['instance_set_numbers'] = instance_set_numbers
save['instance_ic_numbers'] = instance_ic_numbers
save['raw_instances'] = votes_vec_instances
save['raw_workers'] = votes_vec_workers
save['raw_votes'] = votes_vec
save['worker_mats'] = cl.worker_mats
save['worker_prior'] = all_priors[0]
save['instance_prior'] = instance_prior
save['num_epoch'] = 1000
save['burn_in'] = 200

# save
with open('ICLabels_test.pkl', 'wb') as f:
    pkl.dump(save, f)
savemat('ICLabels_test.mat', save, oned_as='column')
