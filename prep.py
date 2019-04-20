import pandas as pd
import pickle
from collections import defaultdict
from operator import sub
import numpy as np
from constant_variable import COLUMNS
from datetime import datetime

''' here are some functions for test'''

import time


def add_list():
    list1 = [1, 2]
    list2 = [3, 4]
    return list1 + list2


lastFM_local_directory = '/Users/shayan/Desktop/Archive/Research/smola_implementation/' \
                         'lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
lastFM_mll_directory = '/home/shayan/Desktop/javad/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
lastFM_saved_Data = './savedData.p'
lastFM_pushe_directory = '/home/shayan/lastfm-dataset-1K/userid-timestamp-artid-artname-traid-traname.tsv'
test_file_address = "./save.p"

'''Constant values'''
current_directory = test_file_address


class LoadAndSaveData:
    def __init__(self, main_data_directory=lastFM_local_directory):
        self.saved_data = "./savedData.csv"
        self.saved_data_pickle = "./savedData.p"
        self.test_data_address = "./save.p"
        self.main_data_directory = main_data_directory

    def save_data(self):
        with open(self.main_data_directory) as my_file:
            data_array = [line.split("\t") for index, line in enumerate(my_file)]
        np.savetxt(self.saved_data, data_array, delimiter=',')

    def save_data_pickle(self):
        with open(self.main_data_directory) as my_file:
            data_array = [line.split("\t") for index, line in enumerate(my_file)]
        with open(self.saved_data_pickle, "wb") as file_for_saving_data:
            pickle.dump(data_array, file_for_saving_data)

    def load_data_pickle(self):
        with open(self.saved_data_pickle, "rb") as test_file:
            data = pickle.load(test_file)
        return data

    def save_data_head(self, head_size=10000):
        """open main last fm file,
           save a test case with the size of
           head_size and dump it as a pickle file """
        with open(self.main_data_directory) as my_file:
            head = [next(my_file).split("\t") for _ in range(head_size)]
        with open(self.test_data_address, "wb") as test_file:
            pickle.dump(head, test_file)

    def load_data_head(self):
        with open(self.test_data_address, "rb") as my_file:
            head = pickle.load(my_file)
        return head

    def load_main_data(self):
        with open(self.saved_data, "rb") as my_file:
            data = pd.read_csv(self.saved_data, delimiter=',')
        return data

    def load_input(self, columns, type="head"):
        """loading input """
        if type == "head":
            unpickled_data = self.load_data_head()
        elif type == "main":
            unpickled_data = self.load_data_pickle()
        else:
            raise TypeError("input type should be either head or main")

        # values = [x.split("\t") for x in unpickled_data]
        # input_file = pd.read_csv(
        #     test_file_address,
        #     encoding= 'utf-8',
        #     sep='\t',
        #     error_bad_lines=False,
        #     names=["userid", "timestamp", "artid", "artname", "traid", "traname"]
        #     )
        # input_file: pd.DataFrame
        input_file = pd.DataFrame(unpickled_data, columns=columns)
        return input_file, unpickled_data

    def save_and_load(self, type="head", head_size=10000):
        if type == "head":
            self.save_data_head(head_size)
            print("type is head")
        elif type == "main":
            self.save_data_pickle()
            print("type is main")
        else:
            raise TypeError("type should be either head or main")
        # input_file: pd.DataFrame
        input_file, _ = self.load_input(COLUMNS, type=type)
        return input_file

    def deprecated_load(self):
        with open(self.main_data_directory) as my_file:
            # input_file = pd.read_csv(
            #     my_file,
            #     encoding='utf-8',
            #     sep='\t',
            #     error_bad_lines=False,
            #     names=COLUMNS)
            input_data = pd.DataFrame(my_file, columns=COLUMNS)
        return input_data


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
    hhmmss = list(map(int, new_time_stamp[1].split(":")[:-1]))
    new_time_stamp_list = yyyymmdd + hhmmss
    return new_time_stamp_list, yyyymmdd, hhmmss, new_time_stamp


def calculate_return_days(input_data, beginning_index, end_index):
    # print("end_index : ", end_index, -"begin_index :", begin_index)
    time_stamps = [calculate_time_stamp(input_data, index)[1:4] for index in range(beginning_index, end_index)]
    # session_exact_time = np.asarray([calculate_return_day(*time_stamp, WEEK)[::2] for time_stamp in time_stamps])
    session_exact_time = np.asarray([calculate_return_day(*time_stamp[:-1], WEEK) for time_stamp in time_stamps])
    return session_exact_time
    # call calculate time stamo on list of indecies


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


def calculage_gap_size_exact(new_time_stamp, old_time_stamp):
    date1 = new_time_stamp[0]
    date2 = old_time_stamp[0]
    time1 = new_time_stamp[1]
    time2 = old_time_stamp[1]
    fmt = '%Y-%m-%d %H:%M:%S'
    d1 = datetime.strptime(date1 + " " + time1[:-1], fmt)
    d2 = datetime.strptime(date2 + " " + time2[:-1], fmt)
    d1_ts = time.mktime(d1.timetuple())
    d2_ts = time.mktime(d2.timetuple())
    diff_in_minute = abs(int(d2_ts - d1_ts) / 60)
    diff_in_minute = round(diff_in_minute,2)
    return diff_in_minute


def calculate_return_day(yyyymmdd, hhmmss, week):
    """Calculate day from return date of user -- Monday is 0 ... Sunday is 6 """
    return_day_number = datetime(yyyymmdd[0], yyyymmdd[1], yyyymmdd[2]).weekday()
    return_day_name = week[return_day_number]
    return_day_hour = hhmmss[0]
    return [return_day_number, return_day_name, return_day_hour]


def preprocess():
    input_data = input_data_value
    """
    @deprecated
     preprocess wrapped_panda_data to make a list of users' sessions
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
        new_time_stamp, yyyymmdd, hhmmss, new_time_stamp_string = calculate_time_stamp(input_data, idx)
        if iteration != 1:
            # Check whether user_ID change or not
            if user_is_same(new_user_number, old_user_number):
                # in every step of the loop idx is reduced by 1.
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
                    """new session condition"""
                    session_actions_list = list(input_data.loc[end_session_index:begin_session_index - 1, COLUMNS[3]])
                    # sessionActionsDict = allActionDict
                    session_times = calculate_return_days(input_data, end_session_index, begin_session_index)

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
                    # print(np.shape(session_times), np.shape(session_tracks))
                    # create output from variables
                    output[new_user_number].append([gap_size, None, session_times, session_tracks])
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


def filter_hole_size(hole_size):
    if hole_size >= GAP_SIZE_MAX_VALUE:
        print("gap size issue")
        gap_size = GAP_SIZE_MAX_VALUE
    elif hole_size > GAP_SIZE_MIN_VALUE:
        gap_size = hole_size
    else:
        gap_size = None
    return gap_size


def new_preprocess():
    input_data = input_data_value
    output = defaultdict(list)
    speed_count = 0  # Check progress
    iteration = ITERATION_START
    begin_session_index = len(input_data) - 1
    print("end", begin_session_index)
    end_time_of_session_index = begin_session_index
    old_time_stamp = None
    for idx in reversed(input_data.index):
        """main loop of program"""
        speed_count += 1
        session_tracks = []
        # Show progress
        """Calculate userID"""
        new_user_number = get_user_id(input_data, idx, COLUMNS)
        new_time_stamp, yyyymmdd, hhmmss, new_time_stamp_string = calculate_time_stamp(input_data, idx)

        if iteration == 1:
            pass
        elif iteration != 1:
            begin_session_index = idx + 1
            # begin session index set to current index + 1
            # since if gap happens we should put begin_index to our list
            #
            if old_time_stamp is not None:
                # hole_size = calculate_gap_size(new_time_stamp, old_time_stamp)
                hole_size_1 = calculage_gap_size_exact(old_time_stamp_string, new_time_stamp_string)
                # print("hole_size : ", hole_size, "hole_size_exact", hole_size_1)
                # print(calculate_return_days(input_data, begin_session_index, begin_session_index + 1))
                # print(new_time_stamp)
                # print(old_time_stamp)
            else:
                raise Exception("Invalid Condition")
            gap_size = filter_hole_size(hole_size_1)
            if hole_size_1 > GAP_SIZE_MIN_VALUE:
                """new session condition"""
                session_actions_list = list(
                    input_data.loc[begin_session_index:end_time_of_session_index - 1, COLUMNS[3]])
                # sessionActionsDict = allActionDict
                session_times = calculate_return_days(input_data, begin_session_index, end_time_of_session_index)

                for item in session_actions_list:
                    session_tracks.append(item)
                end_time_of_session_index = begin_session_index
                # print(gap_size, session_times, session_tracks)
                output[new_user_number].append([gap_size, None, session_times, session_tracks])
            elif gap_size is None:
                # in session condition
                pass
            else:
                raise Exception("Invalid condition")

        else:
            raise Exception("Invalid condition")

        old_time_stamp = new_time_stamp
        old_time_stamp_string = new_time_stamp_string
        iteration += 1
    return output


def reverse_dictionary(preprocess_output):
    new_dictionary = {}
    for key, value in preprocess_output.items():
        new_dictionary[key] = reverse_list(value)
    return new_dictionary

def reverse_list(list_of_elements):
    return list(reversed(list_of_elements))

""" Script for preprocess and dumping wrapped_panda_data with name of shayanPrep"""
# output = preprocess()
# with open('shayanPrep.pk', 'wb') as fi:
#     # dump your wrapped_panda_data into the file
#     pickle.dump(output, fi)

if __name__ == "__main__":
    COLUMNS = ["userid", "timestamp", "artid", "artname", "traid", "traname"]
    input_data_value = load_server_input(COLUMNS, lastFM_mll_directory)
    ITERATION_START = 1
    GAP_THRESHOLD = 19
    NUMBER_OF_INPUT = len(input_data_value)  # len(inputFile.index) = 19098862
    GAP_SIZE_MAX_VALUE = 180 * 24 * 60  # GAP SIZE MAX
    GAP_SIZE_MIN_VALUE = 60  # GAP SIZE MIN VALUE
    WEEK = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
    output = new_preprocess()
    CURRENT_OUTPUT_NAME = 'WithSessionTime.pk'
    with open(CURRENT_OUTPUT_NAME, 'wb') as fi:
        pickle.dump(output, fi)
    new_dict = reverse_dictionary(output)
    CURRENT_OUTPUT_NAME = 'WithSessionTime1.pk'
    with open(CURRENT_OUTPUT_NAME, 'wb') as fi:
        pickle.dump(output, fi)

