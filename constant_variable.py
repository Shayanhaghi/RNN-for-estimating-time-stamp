import pandas as pd
import tensorflow as tf
COLUMNS = ["userid", "timestamp", "artid", "artname", "traid", "traname"]

lastFM_local_directory = '/Users/shayan/Desktop/Archive/Research/smola_implementation/' \
                         'lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
lastFM_mll_directory = '/home/shayan/Desktop/javad/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
lastFM_saved_Data = './savedData.p'
lastFM_pushe_directory = '/home/shayan/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
test_file_address = "./save.p"

def load_server_input(columns, directory):
    """load input in server """
    with open(directory) as my_file:
        input_file = pd.read_csv(
            my_file,
            encoding='utf-8',
            sep='\t',
            error_bad_lines=False,
            names=columns)
    return input_file


def config_tensorboard(session):
    """make a writer"""
    log_directory = "/home/shayan/shayancode/tensorboard_directory"
    writer = tf.summary.FileWriter(log_directory)
    writer.add_graph(session.graph)
    return writer