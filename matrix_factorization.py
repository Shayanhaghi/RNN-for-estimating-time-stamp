# tasks
# order results of different architectures
# poisson cost
#
# multi layer + dilation

# check whether memory is working proper or not
#
# create matrix
#  which it's row is users
# it's columns are music_id or album_id or artist_id
#
import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
import pickle
# from ConstantVariables import lastFM_mll_directory
# from prep import LoadAndSaveData
COLUMNS = ["userid", "timestamp", "artid", "artname", "traid", "traname"]


def load_server_input(columns, directory):
    """load server inputs """
    with open(directory) as my_file:
        input_file = pd.read_csv(
            my_file,
            encoding='utf-8',
            sep='\t',
            error_bad_lines=False,
            names=columns)
    return input_file


def pickle_save(name, data):
    with open(name, "wb") as test_file:
        pickle.dump(data, test_file)


def pickle_load(name):
    with open(name, "rb") as test_file:
        data = pickle.load(test_file)
    return data


def factorize(user_item_matrix, number_of_components=100):
    nmf = NMF(n_components=number_of_components)
    user_decomposed = nmf.fit_transform(user_item_matrix)
    item_decomposed = nmf.components_
    print(user_decomposed.shape)
    print(item_decomposed.shape)
    return user_decomposed, item_decomposed


def make_matrix(panda_data: pd.DataFrame):
    art_name_list = list(set(panda_data.loc[:, COLUMNS[3]]))
    user_id_list = list(set(panda_data.loc[:, COLUMNS[0]]))
    art2number_dict, number2art_dict, user2number_dict, number2user_dict = make_dictionaries(art_name_list,
                                                                                             user_id_list)
    number_of_users = len(user_id_list)
    number_of_items = len(art_name_list)

    input_matrix_2factorize = np.zeros((number_of_users, number_of_items))

    for data_row_index in range(len(panda_data)):
        single_row_art_name = panda_data.loc[data_row_index, COLUMNS[3]]
        single_row_user_id = panda_data.loc[data_row_index, COLUMNS[0]]
        user_id_number = user2number_dict[single_row_user_id]
        art_name_number = art2number_dict[single_row_art_name]
        input_matrix_2factorize[user_id_number][art_name_number] = \
            input_matrix_2factorize[user_id_number][art_name_number] + 1

    return input_matrix_2factorize, art2number_dict, number2art_dict, user2number_dict, number2user_dict


def make_dictionaries(art_name_list, user_id_list):
    art2number_dict = {}
    number2art_dict = {}
    user2number_dict = {}
    number2user_dict = {}
    for number, art in enumerate(art_name_list):
        art2number_dict[art] = number
        number2art_dict[number] = art
    for number, user in enumerate(user_id_list):
        user2number_dict[user] = number
        number2user_dict[number] = user

    return art2number_dict, number2art_dict, user2number_dict, number2user_dict


if __name__ == "__main__":
    # loadAndSaveData = LoadAndSaveData(lastFM_mll_directory)
    # wrapped_panda_data = loadAndSaveData.save_and_load(type="main")
    # user_item_count_matrix, art2number_dict, number2art_dict, user2number_dict,
    # number2user_dict = make_matrix(wrapped_panda_data)
    user_item_count_matrix = np.load("user_item_count")
    user_embedding_matrix, art_embedding_matrix = factorize(user_item_count_matrix, number_of_components=100)
    np.save("user_embedding_matrix.npy", user_embedding_matrix)
    np.save("art_embedding_matrix.npy", art_embedding_matrix)
    t1 = np.load("user_embedding_matrix.npy")
    t2 = np.load("art_embedding_matrix.npy")
    # np.load("user_item_count")
    # pickle_save("art2number_dict", art2number_dict)
    # pickle_save("number2art_dict", number2art_dict)
    # pickle_save("user2number_dict", user2number_dict)
    # pickle_save("number2user_dict", number2user_dict)
    # pickle_save("user_item_count", user_item_count_matrix)
# snippet for making user_id_set adn art_id_set
# for data_row in range(len(wrapped_panda_data)):
#     user_id_set.add(wrapped_panda_data.loc[data_row, COLUMNS[0]])
#     print(len(art_id_set))
#     art_id_set.add(wrapped_panda_data.loc[data_row, COLUMNS[3]])
# print(len(artNameList), list(artNameList)[0:3], sep="***\n")
# print(len(userIdList), list(userIdList)[0:3], sep="***\n")

# x = dict(list(art_id_set))
# print(x[0])
