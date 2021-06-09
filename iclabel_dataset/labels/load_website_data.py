import sqlite3 as sql
import numpy as np
import pdb


def load_icl(db_path):

    # load sqlite data
    connection = sql.connect(db_path)
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM users')
    db_combined = cursor.fetchall()
    cursor.execute('SELECT * FROM labels')
    db_labels = cursor.fetchall()
    db_labels_column_names = [x[0] for x in cursor.description]
    cursor.execute('SELECT * FROM images')
    db_images = cursor.fetchall()
    connection.close()
    del connection, cursor

    # remove users with not enough labels
    min_labels = 10
    user_labs = [x[1] for x in db_labels]
    user_labs_count = np.array([user_labs.count(x) for x in [x[0] for x in db_combined]])
    keep_users = np.where(user_labs_count >= min_labels)[0]
    db_combined = [db_combined[x] for x in keep_users]
    del user_labs_count

    # remove labels from users with not enough labels
    db_labels = [x for x in db_labels if x[1] in [y[0] for y in db_combined]]
    del keep_users

    # remove instances which only have "?" as an answer
    #   find all images with a ?
    #   for each of those images, find all labels
    #   if the labels are only ?, remove
    remove = list()
    for it in np.unique([x[2] for x in db_labels if x[10]]):
        if not np.sum([x[3:10] for x in db_labels if x[2] == it]):
            remove.append(it)
    if remove:
        db_labels = [x for x in db_labels if x[2] not in remove]
        NotImplementedError('there are some dead answers that need input')

    # TODO: fix the above. doesn't catch everything

    # aggregate images
    db_images = [db_images[y-1] for y in np.unique([x[2] for x in db_labels])]

    # tabulate data
    I = len(set([x[2] for x in db_labels]))  # number of images
    A = len(db_combined)  # number of users and experts combined

    # dictionary for all
    combined_ind = [x[0] for x in db_combined]
    combined_dict = {x: y for x, y in zip(combined_ind, range(A))}  # sqlite index to db_experts index

    # dictionary for images
    im_ind = list(set([x[2] for x in db_labels]))
    im_ind.sort()
    im_dict = {x: y for x, y in zip(im_ind, range(I))}  # sqlite image_id to image index

    # separate votes_mat
    votes_mat = np.array([x[3:11] for x in db_labels])
    is_expert = np.array([x[4] for x in db_combined])
    # is_expert[0] = 0

    # index votes_mat
    iV = np.array([im_dict[x[2]] for x in db_labels])
    uV = np.array([combined_dict[x[1]] for x in db_labels])

    # tabulate more data
    V = len(votes_mat)  # number of total votes_mat
    T = 7  # number of topics (estimated truth)
    C = T + 1  # number of categories (options for voting)

    # reshape votes_mat
    nz = np.nonzero(votes_mat)
    votes_vec = nz[1]
    votes_vec_workers = uV[nz[0]]
    votes_vec_instances = iV[nz[0]]
    VV = len(votes_vec)

    # dataset info
    instance_set_numbers = np.array([x[2] for x in db_images])
    instance_ic_numbers = np.array([x[3] for x in db_images])
    instance_ids = np.array([x[0] for x in db_images])

    return {'votes': votes_vec,
            'workers': votes_vec_workers,
            'instances': votes_vec_instances,
            'is_expert': is_expert,
            'instance_set_numbers': instance_set_numbers,
            'instance_ic_numbers': instance_ic_numbers,
            'instance_ids': instance_ids,
            'worker_ids': np.array([x[1] for x in db_combined]),
            'vote_ids': np.array(db_labels_column_names[3:11]),
            'n_votes': V,
            'n_classes': T,
            'n_responses': C,
            'n_instances': I,
            'n_workers': A}
