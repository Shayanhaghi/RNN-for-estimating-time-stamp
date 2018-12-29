import pandas as pd
import tensorflow as tf
import pickle
from collections import defaultdict
import csv
from operator import sub
import datetime

''' here are some functions for test'''


def add_list():
    list1 = [1, 2]
    list2 = [3, 4]
    return list1 + list2


local_directory = '/Users/shayan/Desktop/Todo/Research/smola_implementation/' \
                  'lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
mll_directory = '/home/shayan/Desktop/javad/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
pushe_directory = '/home/shayan/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
test_file_dir = "./save.p"

'''Constant values'''
current_directory = test_file_dir


class LoadHeadTest():
    def __init__(self):
        pass

    @staticmethod
    def save_data_head(self):
        """save head of the data which is load from Last_fm in """
        with open(local_directory) as my_file:
            head = [next(my_file) for _ in range(100)]
        with open("./save.p", "wb") as my_file:
            pickle.dump(head, my_file)

    @staticmethod
    def load_data_head():
        with open("./save.p", "rb") as my_file:
            head = pickle.load(my_file)
        return head

    def load_input(self, columns):
        """loading input head"""
        unpickled_data = self.load_data_head()
        values = [x.split("\t") for x in unpickled_data]
        # input_file = pd.read_csv(
        #     test_file_dir,
        #     encoding= 'utf-8',
        #     sep='\t',
        #     error_bad_lines=False,
        #     names=["userid", "timestamp", "artid", "artname", "traid", "traname"]
        #     )
        input_file = pd.DataFrame(values, columns=columns)
        return input_file


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


def calculate_time_stamp(input_data, idx):
    """Calculate current Timestamp time loaded and extracted to array of year, month, week, hours, minutes, seconds"""
    new_time_stamp = input_data.loc[idx, "timestamp"].split("T")
    yyyymmdd = list(map(int, new_time_stamp[0].split("-")))
    hhmmss = list(map(int, new_time_stamp[1].split(":")[0:2]))
    new_time_stamp = yyyymmdd + hhmmss
    return new_time_stamp, yyyymmdd, hhmmss


def get_user_id(input_data, index, columns):
    """change current user id index to correct one"""
    user_id = input_data.loc[index, columns[0]]
    new_user_number = int(user_id.split('_')[1])
    return new_user_number


def user_is_same(current_user_number, past_user_numbers):
    """Check whether current user number and past are same"""
    if current_user_number == past_user_numbers:
        return True
    else:
        return False


def calculate_gap_size(new_time_stamp, old_time_stamp):
    """calculate gap size from two successive time stamp"""
    diff = list(map(sub, new_time_stamp, old_time_stamp))
    hole_size = diff[0] * 365 * 24 * 60 + diff[1] * 30 * 24 * 60 + diff[2] * 24 * 60 + diff[3] * 60 + diff[4]
    return hole_size


def calculate_return_day(yyyymmdd, hhmmss, week):
    """Calculate day from return date of user -- Monday is 0 ... Sunday is 6 """
    return_day_number = datetime.datetime(yyyymmdd[0], yyyymmdd[1], yyyymmdd[2]).weekday()
    return_day_name = week[return_day_number]
    return_day_hour = hhmmss[0]
    return [return_day_number, return_day_name, return_day_hour]


COLUMNS = ["userid", "timestamp", "artid", "artname", "traid", "traname"]
input_data = load_server_input(COLUMNS, mll_directory)
ITERATION_START = 1
GAP_THRESHOLD = 19
NUMBER_OF_INPUT = len(input_data)  # len(inputFile.index) = 19098862
GAP_SIZE_MAX_VALUE = 200 * 24 * 60  # GAP SIZE MAX
GAP_SIZE_MIN_VALUE = 61  # GAP SIZE MIN VALUE
WEEK = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}


def test_empty_session():
    for i in range(len(input_data)):
        if input_data.loc[i, "userid"] == "user_000283":
            print(input_data.loc[i, :])


def preprocess():
    """ preprocess data to make a list of users' sessions
        user -> list of sessions
        session -> [time,[list of art_id elements],beginning of session, end of session],[]]
    """
    output = defaultdict(list)
    # print(input_data.loc[0:2, "artname"])
    print(input_data.loc[1:10, "userid"])
    allActionList = list(set(input_data.loc[:, "artname"]))  # list of art names

    speed_count = 0  # Check progress
    iteration = ITERATION_START
    end_session_index = len(input_data) - 1
    print("end", end_session_index)
    begin_session_index = end_session_index

    for idx in reversed(input_data.index):
        """main loop of program"""
        speed_count += 1
        session_tracks = []
        # Show progress
        """Calculate userID"""
        # userId = input_data.loc[idx, "userid"]
        # new_user_number = int(userId.split('_')[1])
        new_user_number = get_user_id(input_data, idx, COLUMNS)
        old_user_number = new_user_number  # initializing old user number to the new
        new_time_stamp, yyyymmdd, hhmmss = calculate_time_stamp(input_data, idx)
        if iteration != 1:
            # Check whether user_ID change or not
            if user_is_same(new_user_number, old_user_number):
                # +1 is for reading dataset from bottom to top
                end_session_index = idx + 1

                hole_size = calculate_gap_size(new_time_stamp, old_time_stamp)
                if hole_size >= GAP_SIZE_MAX_VALUE:
                    print("gap size issue")
                    print("size: ", hole_size, old_user_number, new_user_number)
                    #  this strategy which limits the gap to a certain maximum value
                    #  can be altered. it depends on the feeding to network mechanism
                    gap_size = GAP_SIZE_MAX_VALUE
                elif hole_size > GAP_SIZE_MIN_VALUE:
                    # Calculate session actions
                    gap_size = hole_size
                if hole_size > GAP_SIZE_MIN_VALUE or hole_size >= GAP_SIZE_MAX_VALUE:
                    session_actions_list = list(input_data.loc[end_session_index:begin_session_index - 1, COLUMNS[3]])
                    # sessionActionsDict = allActionDict
                    for item in session_actions_list:
                        session_tracks.append(item)

                    '''
                    if we need time of beginning and ending of current session following 
                    code can be used.
                    '''
                    #  begin_session_time_stamp = calculate_time_stamp(input_data, begin_session_index)
                    #  end_session_time_stamp = calculate_time_stamp(input_data, end_session_index)

                    begin_session_index = end_session_index
                    # calculate day number, name, hour
                    return_day_number, return_day_name, return_day_hour = calculate_return_day(yyyymmdd, hhmmss, WEEK)
                    # create output from variables
                    output[new_user_number].append([gap_size, return_day_name, return_day_hour, session_tracks])
                    # output[new_user_number].append(
                    # [gap_size, return_day_name, return_day_hour, sessionTracks, begin_session_timestamp,
                else:
                    # in session condition
                    pass
            elif not user_is_same(new_user_number, old_user_number):
                """if user is changed we ignore previous sequence and set both begin and end of 
                   session to same value which is current index """
                begin_session_index = idx
                end_session_index = idx

            else:
                raise (Exception("invalid condition"))

        else:
            """iteration-1 case """
            # for now we just ignore 1st time stamp
            pass

        old_time_stamp = new_time_stamp
        old_user_number = new_user_number
        iteration += 1
    return output


""" Script for preprocess and dumping data with name of shayanPrep"""
# output = preprocess()
# with open('shayanPrep.pk', 'wb') as fi:
#     # dump your data into the file
#     pickle.dump(output, fi)

test_empty_session()
