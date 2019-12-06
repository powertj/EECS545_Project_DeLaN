import random

def random_train_test_chars(data, num_train_chars=1, num_samples_per_char=1):
    '''
    Use with cartpole dataset. Picks a specified number of train characters at random and evaluates on the rest.
    '''
    char_count = {}
    char_indices = {}
    train_labels = []
    test_labels = []
    train_chars = []
    test_chars = []
    train_trajectories = []
    test_trajectories = []

    for i, label in enumerate(data['labels']):
        idx = label[0]
        letter = data['keys'][idx-1][0]
        
        if letter in char_count:
            char_count[letter] += 1
            char_indices[letter].append(i)

        else:
            test_chars.append(letter)
            char_count[letter] = 1
            char_indices[letter] = [i]

    for i in range(num_train_chars):
        if len(test_chars) > 0:
            train_char_idx = random.randint(0,len(test_chars)-1)
            train_char = test_chars.pop(train_char_idx)
            train_chars.append(train_char)
            if num_samples_per_char < len(char_indices[train_char]):
                train_trajectories += char_indices[train_char][:num_samples_per_char]
                train_labels += [train_char] * num_samples_per_char

            else:
                train_trajectories += char_indices[train_char]
                train_labels += [train_char] * len(char_indices[train_char])

    for test_char in test_chars:
            if num_samples_per_char < len(char_indices[test_char]):
                test_trajectories += char_indices[test_char][:num_samples_per_char]
                test_labels += [test_char] * num_samples_per_char

            else:
                test_trajectories += char_indices[test_char]
                test_labels += [test_char] * len(char_indices[test_char])

    return train_trajectories, train_labels, test_trajectories, test_labels

def random_train_test_trajectories(data, num_train_labels=1, num_samples_per_label=1):
    '''
    Use with cartpole dataset. Picks a specified number of train labels at random and evaluates on the rest.
    '''
    label_count = {}
    label_indices = {}
    train_labels = []
    test_labels = []
    test_label_types = []
    train_trajectories = []
    test_trajectories = []

    for i, label in enumerate(data['labels'].flatten()):
        if label in label_count:
            label_count[label] += 1
            label_indices[label].append(i)

        else:
            test_label_types.append(label)
            label_count[label] = 1
            label_indices[label] = [i]

    for i in range(num_train_labels):
        if len(test_label_types) > 0:
            train_label_idx = random.randint(0,len(test_label_types)-1)
            train_label = test_label_types.pop(train_label_idx)
            train_labels.append(train_label)
            if num_samples_per_label < len(label_indices[train_label]):
                train_trajectories += label_indices[train_label][:num_samples_per_label]
                train_labels += [train_label] * num_samples_per_label

            else:
                train_trajectories += label_indices[train_label]
                train_labels += [train_label] * len(label_indices[train_label])


    for test_label in test_label_types:
            if num_samples_per_label < len(label_indices[test_label]):
                test_trajectories += label_indices[test_label][:num_samples_per_label]
                test_labels += [test_label] * num_samples_per_label

            else:
                test_trajectories += label_indices[test_label]
                test_labels += [test_label] * len(label_indices[test_label])


    return train_trajectories, train_labels, test_trajectories, test_labels

def select_train_test_trajectories(data, train_label_types=[1], num_samples_per_label=1):
    '''
    Use with cartpole dataset. Specify which labels to train on and evaluates on the rest.
    '''
    label_count = {}
    label_indices = {}
    train_labels = []
    test_labels = []
    test_label_types = []
    train_trajectories = []
    test_trajectories = []

    for i, label in enumerate(data['labels'].flatten()):
        if label in label_count:
            label_count[label] += 1
            label_indices[label].append(i)

        else:
            test_label_types.append(label)
            label_count[label] = 1
            label_indices[label] = [i]

    for train_label in train_label_types:
            if train_label in test_label_types:
                test_label_types.remove(train_label)
            if num_samples_per_label < len(label_indices[train_label]):
                train_trajectories += label_indices[train_label][:num_samples_per_label]
                train_labels += [train_label] * num_samples_per_label

            else:
                train_trajectories += label_indices[train_label]
                train_labels += [train_label] * len(label_indices[train_label])


    for test_label in test_label_types:
            if num_samples_per_label < len(label_indices[test_label]):
                test_trajectories += label_indices[test_label][:num_samples_per_label]
                test_labels += [test_label] * num_samples_per_label

            else:
                test_trajectories += label_indices[test_label]
                test_labels += [test_label] * len(label_indices[test_label])

    return train_trajectories, train_labels, test_trajectories, test_labels

# def generate_one_shot_train_test_indices(data, train_labels, test_label, num_samples_per_label=1):
#     label_count = {}
#     label_indices = {}
#     train_trajectories = []
#     test_trajectories = []

#     for i, label in enumerate(data['labels'].flatten()):
#         if label in label_count:
#             label_count[label] += 1
#             label_indices[label].append(i)

#         else:
#             label_count[label] = 1
#             label_indices[label] = [i]

#     for train_label in train_labels:
#             if num_samples_per_label < len(label_indices[train_label]):
#                 train_trajectories += label_indices[train_label][:num_samples_per_label]
#             else:
#                 train_trajectories += label_indices[train_label][1:]
    
#     train_trajectories += [label_indices[test_label][0]]

#     if num_samples_per_label < len(label_indices[test_label]):
#         test_trajectories += label_indices[test_label][1:num_samples_per_label+1]
#     else:
#         test_trajectories += label_indices[test_label][1:]

#     return train_trajectories, test_trajectories