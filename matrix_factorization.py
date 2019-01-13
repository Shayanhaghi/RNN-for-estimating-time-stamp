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
from prep import LoadHeadTest

COLUMNS = ["userid", "timestamp", "artid", "artname", "traid", "traname"]
mll_directory = '/home/shayan/Desktop/javad/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'


def load_server_input(columns, directory):
    "load input in server "
    """load server inputs """
    with open(directory) as my_file:
        input_file = pd.read_csv(
            my_file,
            encoding='utf-8',
            sep='\t',
            error_bad_lines=False,
            names=columns)
    return input_file


number_of_users = 1000
number_of_items = 10000
factorized_matrix_placeholder = np.zeros((number_of_users, number_of_items))
loadHeadTest = LoadHeadTest()
# loadHeadTest.save_data_head()
data = loadHeadTest.load_input(COLUMNS)
user_id_set = set()
art_id_set = set()
print(data.loc[0, COLUMNS[0]])
print(data.loc[1, COLUMNS[3]])


def makeMatrix(factorized_matrix_placeholder: np.ndarray, data: pd.DataFrame):
    for i in range(len(data)):

