import tensorflow as tf
import pandas as pd


class PreProcessing:
    def __init__(self):
        self.addressToReadData = None
        typeIsLocal = True
        self.set_directory(typeIsLocal)
        self.dataFrame = None
        self.read_data_with_pandas()

    # set directory to read wrapped_panda_data from
    def set_directory(self, local=False, pushe=True):
        pushe_directory = '/home/shayan/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
        server_directory = '/home/shayan/Desktop/javad/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
        local_directory = '/Users/shayan/Desktop/Todo/Research/smola_implementation/' \
                          'lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
        if local:
            self.addressToReadData = local_directory
        else:
            self.addressToReadData = server_directory
        if pushe:
            self.addressToReadData = pushe_directory

    def read_data_with_pandas(self):
        self.dataFrame = pd.read_csv(self.addressToReadData, engine='python', sep='\t',
                                     error_bad_lines=False,
                                     names=["userid", "timestamp", "artid", "artname", "traid", "traname"])


prep = PreProcessing()


print(len(set(prep.dataFrame.userid)))


