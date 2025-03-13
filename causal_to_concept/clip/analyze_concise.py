
import sklearn.linear_model
import numpy as np
import h5py
import csv
import os


def logistic_regression(x1, x2, solver='lbfgs'):
    labels = np.concatenate([np.array(len(x1) * [0]), np.array(len(x2) * [1])])
    all_x = np.concatenate([x1, x2], axis=0)
    clf = sklearn.linear_model.LogisticRegression(solver=solver, max_iter=250)
    clf.fit(all_x, labels)
    vector = clf.coef_.flatten()
    return vector


def get_v(v_list):
    for i in range(10):
        v_init = sum(v_list)
        for j in range(len(v_list)):
            if np.dot(v_init, v_list[j]) < 0:
                v_list[j] = - v_list[j]
    v = sum(v_list) / len(v_list)
    return v


def split_dataset(embeddings, labels, var):
    vars = [0, 1, 2, 3, 4, 5]
    vars.remove(var)
    all_indices = np.arange(len(embeddings))
    for v in vars:
        values = np.unique(labels[:, v])
        values = values[:(len(values)// 2)]
        indices_val = np.where(np.isin(labels[:, v], list(values)))
        all_indices = np.intersect1d(all_indices, indices_val)
    # print(len(all_indices))
    rem_indices = np.arange(len(embeddings))
    rem_indices = np.delete(rem_indices, np.where(np.isin(rem_indices, all_indices)))
    # print(len(rem_indices))
    return all_indices, rem_indices


for model in ['vit', 'rn']:
    train_ratio = .5

    if model == 'vit':
        dir = './data_transformed'
    elif model == 'rn':
        dir = './data_transformed_rn'
    else:
        raise ValueError

    dataset = h5py.File('3dshapes.h5', 'r')
    images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
    labels = dataset['labels']  # array shape [480000,6], float64

    # 0 is cube
    # 1 is cylinder
    # 2 is ball
    # 3 is ellipsoid
    image_shape = images.shape[1:]  # [64,64,3]
    label_shape = labels.shape[1:]  # [6]
    n_samples = labels.shape[0]  # 10*10*10*8*4*15=480000

    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}

    embeddings = np.load(os.path.join(dir, 'all_embeddings.npy'))
    # print(np.std(embeddings))
    # print(np.std(embeddings, axis=0)[:10])
    anal_vars = [0, 1, 2, 3, 4, 5]
    # anal_vars = [4]
    eps = .001
    for index in anal_vars:
        print(_FACTORS_IN_ORDER[index])
        values = np.unique(labels[:, index])
        n_values = len(values)
        embedding_list = []
        embedding_list_test = []
        label_list = []
        for v in values:
            indices = np.where((labels[:, index] >= v - eps) & (labels[:, index] <= v + eps))
            embeddings_value = embeddings[indices]
            if train_ratio is not None:
                train_data, test_data = split_dataset(embeddings_value, labels[indices], index)

                embedding_list.append(embeddings_value[train_data])
                embedding_list_test.append(embeddings_value[test_data])
                label_list.append(labels[indices])
            else:
                embedding_list_test.append(embeddings_value)
                embedding_list.append(embeddings_value)
        # dirs collects the directions of the pairwise logistic regression
        dirs = []
        for i in range(min(n_values, 50)):
            for j in range(i):
                dirs.append(logistic_regression(embedding_list[i], embedding_list[j]))

        v = get_v(dirs)

        np.save(os.path.join(dir, _FACTORS_IN_ORDER[index] + '.npy'), v)
        v = np.expand_dims(v, axis=0)
        rows = [['Value', 'Mean valuation', 'Std valuation']]
        for value, emb in zip(values, embedding_list_test):
            valuations = np.sum(v * emb, axis=1)
            rows.append([value, np.mean(valuations), np.std(valuations)])

        file_path = os.path.join(dir, _FACTORS_IN_ORDER[index] + '.csv')

        # Write rows to CSV file
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)
