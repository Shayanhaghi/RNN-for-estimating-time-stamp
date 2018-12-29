import pandas as pd


COLUMNS = ["userid", "timestamp", "artid", "artname", "traid", "traname"]





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
