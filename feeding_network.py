import numpy as np
import pickle
from tensorboardX import SummaryWriter
import tensorflow as tf
import copy
import random

data_address_pushe = "/home/shayan/simpleRNN/shayanPrep"
data_address_mll = "/home/shayan/shayancode/shayanPrep.pk"
data_address_mll2 = "/home/shayan/shayancode/newPrep.pk"
directory_address = "/home/shayan/shayancode/"
from constant_variable import config_tensorboard

""" some untold defaults :
    the name "feeding_values " is used as a output of ExtractInfo Class.
    As an input 

"""


class ExtractInfo:
    """ extract user, and number of items in each session  from wrapped_panda_data"""

    def __init__(self):
        """init  values"""
        self.main_data = None
        self.max_user_number = 1000
        self.session_type_number = 1
        # 1 for the gap
        self.input_dimension = self.session_type_number + 1
        self.data_address = data_address_mll2
        self.min_acceptable_sequence_length = 16 * 32 + 32

    def load_data(self):
        """save measures"""
        print(self.data_address)
        print("start loading data")
        with open(self.data_address, "rb") as data:
            self.main_data = pickle.load(data)
        print("wrapped_panda_data has been loaded.")

    def calculate_measures(self):
        """calculate wrapped_panda_data 2 feed to the network"""
        print(len(self.main_data))
        for i in range(1001):
            if self.main_data[i] == []:
                print(i)
        user_session = []
        invalide_users = []
        counter = 0
        for user_index, user_data in self.main_data.items():
            user_session_matrix = np.zeros([len(user_data), self.input_dimension + 1])
            for index, session_data in enumerate(user_data):
                # session wrapped_panda_data[0] is gap size, session_data[3] is item's list
                # TODO  for multiple session this code should be changed
                user_session_matrix[index, 0] = user_index
                user_session_matrix[index, 1] = session_data[0]
                user_session_matrix[index, 2] = len(session_data[3])
            if user_session_matrix.shape[0] > self.min_acceptable_sequence_length:
                counter = counter + 1
                # print(user_session_matrix.shape[0])
                user_session.append(user_session_matrix)
            else:
                invalide_users.append(user_index, )
            # print(user_session[counter][1:5, 0:3])

        print()
        return user_session

    @staticmethod
    def save_values(data):
        with open("feeding_values", "wb") as file:
            pickle.dump(data, file)

    def extract_and_save(self):
        self.load_data()
        user_sessions = self.calculate_measures()
        self.save_values(user_sessions)


class ExtendInfo(ExtractInfo):
    def __init__(self):
        """init  values"""
        super().__init__()
        self.main_data = None
        self.max_user_number = 1000
        self.input_dimension = 4
        self.data_address = "/home/shayan/shayancode/WithSessionTime1.pk"
        self.extract_and_save()

    @staticmethod
    def extract_session_time_begin_mid_end(session_values):
        day, hour = np.mean(session_values, axis=0)
        return day, hour

    def calculate_measures(self):
        """calculate wrapped_panda_data 2 feed to the network"""
        print(len(self.main_data))
        for i in range(1001):
            if self.main_data[i] == []:
                print(i)
        user_sessions = []
        counter = 0
        for user_index, user_data in self.main_data.items():
            user_session_matrix = np.zeros([len(user_data), self.input_dimension + 1])
            for index, session_data in enumerate(user_data):
                # session wrapped_panda_data[0] is gap size, session_data[3] is item's list
                # TODO  for multiple session this code should be changed
                user_session_matrix[index, 0] = user_index
                user_session_matrix[index, 1] = session_data[0]
                user_session_matrix[index, 2] = len(session_data[3])
                exact_times = session_data[2][:, ::2]
                exact_times = np.array(exact_times).astype(np.int)
                day, hour = self.extract_session_time_begin_mid_end(exact_times)
                user_session_matrix[index, 3] = day
                user_session_matrix[index, 4] = hour
                """user_session_matrix[index,3]
                   is a list of sessions of users, 
                   each session is """
                # we should read this in next phase
            if user_session_matrix.shape[0] > self.min_acceptable_sequence_length:
                counter = counter + 1
                user_sessions.append(user_session_matrix)
            else:
                print(user_index, user_session_matrix.shape[0])
            # print(user_session[counter][1:5, 0:3])
            print(counter)
        return user_sessions


class DataAnalysis(ExtractInfo):

    def load_data_with_tests(self):
        """save measures"""
        print("start loading wrapped_panda_data")
        with open(self.data_address, "rb") as data:
            self.main_data = pickle.load(data)
            print(len(self.main_data))
            # 992 -> number 0f users
            print(len(self.main_data[1]))
            # number of session of number 1 user -> 907

            print(self.main_data[1][1])
            # [1133, 'Wednesday', 10, ['Jazztronik',
            # 'Maximum Style & Jb Rose', 'Silent Poets',
            # 'Sleepwalker', 'Roni Size & Reprazent', 'Soul Dhamma',
            #  'Rinôçérôse', 'Goldie', 'Herbie Hancock', 'Jimmy Edgar',
            #  'Bayaka', 'Tiga', 'Soundgarden', 'Das Bierbeben',
            #  'Final Drop', 'Mine', 'Kyoto Jazz Massive',
            # 'Boom Boom Satellites', 'Roni Size & Reprazent',
            # 'Prefuse 73', 'Chari Chari', 'Chari Chari',
            # 'Chari Chari', 'Chari Chari', 'Chari Chari',
            # 'Chari Chari', 'Chari Chari', 'Chari Chari',
            # 'Chari Chari', 'Chari Chari', 'Chari Chari',
            # 'Chari Chari']]

            # type of self.main_data[1][1]
            # [gap in minute unit, day of week, hour of day, list of items in session]

            print(type(self.main_data))
            # collections.defaultdict
        print("wrapped_panda_data has been loaded.")

    def load_data(self):
        with open(self.data_address, "rb") as data:
            self.main_data = pickle.load(data)
        print("wrapped_panda_data has been loaded.")

    def visualize_session_length_histogram(self):
        session_number = list()
        for user_number in range(len(self.main_data)):
            session_number.append(len(self.main_data[user_number]))
        session_number_1 = np.asarray(session_number)
        session_number_1 = np.float32(session_number_1)
        session_number_tensor = tf.convert_to_tensor(session_number_1)
        del session_number
        print(session_number_tensor)
        hist = tf.summary.histogram("number of sessions histogram",
                                    session_number_tensor)
        test = tf.placeholder(tf.float32, [1000, 1], "test_var")
        feed_value = np.ones([1000, 1], np.float32)
        mergedSummary = tf.summary.merge_all()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        writer = config_tensorboard(sess)
        for i in range(2):
            summary = sess.run(mergedSummary, {test: feed_value})
            writer.add_summary(summary, i)

    def visualize_gap_visualization(self):
        gap_length_values = list()
        max_value = 0
        for user_number in range(len(self.main_data)):
            for user_session in self.main_data[user_number]:
                gap_length_values.append(np.round(user_session[0] / 60.0))
                max_value = max(np.round(user_session[0] / 60.0), max_value)
                # print(np.round(user_session[0]/60.0))
        print("max_Value", max_value)
        gap_length = np.asarray(gap_length_values)
        gap_length = np.float32(gap_length)
        gap_length = tf.convert_to_tensor(gap_length)
        del gap_length_values
        for i in range(20):
            print(gap_length)
            print("1000")
        hist = tf.summary.histogram("number of sessions histogram",
                                    gap_length)
        test = tf.placeholder(tf.float32, [1000, 1], "test_var")
        feed_value = np.ones([1000, 1], np.float32)
        mergedSummary = tf.summary.merge_all()
        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())
        writer = config_tensorboard(sess)
        for i in range(2):
            summary = sess.run(mergedSummary, {test: feed_value})
            writer.add_summary(summary, i)


class BatchFeeder:
    """Feeding batch to network"""

    def __init__(self):
        self.UNROLL_SIZE = 16
        self.data = []
        self.user_number = 0
        self.gap_vector_size = 16
        self.load_data()
        self.users_feed_index = np.zeros([1000], int)
        # user current number of index of start of the time
        self.user_max = np.zeros([1000], int)

        self.user_current_test_data_index = np.zeros([1000], int)
        # maximum valid time index for wrapped_panda_data for each user
        self.user_train_max = np.zeros([1000], int)
        #
        self.train_test_split_rate = 0.8
        # if train_test_split = 0.8 : end 20% of wrapped_panda_data used for test
        # and 80% of first part of wrapped_panda_data is used for training

        self.batch_size = 64
        # batch_size of the network
        self.dataSize = len(self.data)
        self.current_selected_users = []
        self.step_in_time = self.UNROLL_SIZE
        self.HIDDDEN_LSTM_SIZE = 80
        self.user_states = (np.zeros([self.batch_size, self.HIDDDEN_LSTM_SIZE]),
                            np.zeros([self.batch_size, self.HIDDDEN_LSTM_SIZE]))

        self.set_user_max_train_max_value()

    """
    def __init__(self, hidden_lstm_size, batch_size, unrolled_size, train_test_split_rate):
        self.UNROLL_SIZE = 16
        self.data = []
        self.user_number = 0
        self.load_data()
        self.users_feed_index = np.zeros([1000], int)
        # user current number of index of start of the time
        self.user_max = np.zeros([1000], int)
        # maximum valid time index for wrapped_panda_data for each user
        self.user_train_max = np.zeros([1000], int)
        #
        self.train_test_split_rate = train_test_split_rate
        # if train_test_split = 0.8 : end 20% of wrapped_panda_data used for test
        # and 80% of first part of wrapped_panda_data is used for training

        self.batch_size = batch_size
        # batch_size of the network
        self.dataSize = len(self.data)
        self.current_selected_users = []
        self.step_in_time = unrolled_size
        self.HIDDDEN_LSTM_SIZE = hidden_lstm_size
        self.user_states = (np.zeros([self.batch_size, self.HIDDDEN_LSTM_SIZE]),
                            np.zeros([self.batch_size, self.HIDDDEN_LSTM_SIZE]))
    """

    def load_data(self, address="feeding_values"):
        """ load wrapped_panda_data """
        with open(address, "rb") as file:
            data = pickle.load(file)
        self.data = data

        # wrapped_panda_data shape :
        self.user_number = len(data)

    def set_user_max_train_max_value(self):
        """ set matrix of max_value and train_max_index for each user """
        for user_index, user_sessions in enumerate(self.data):
            self.user_max[user_index] = int(user_sessions.shape[0])
            self.user_train_max[user_index] = int(user_sessions.shape[0] * 0.8)
            # print(self.user_max[1:10])
            # print(self.user_train_max[1:10])

    def choose_users(self):
        """self.data size is very important """
        self.current_selected_users = np.random.randint(0, self.dataSize, self.batch_size)
        # print(self.current_selected_users)

    def check_user_data(self):
        # print(self.data[user_number][0][0] for user_number in self.current_selected_users)
        print(self.data[0:10][1][0])
        print(self.current_selected_users)

    def move_forward_current_users_index(self):
        for user in set(self.current_selected_users):
            print("", set(self.current_selected_users))
            self.users_feed_index[user] = self.users_feed_index[user] + self.step_in_time

    def check_user_is_finished(self):
        for user in self.current_selected_users:
            # if we reach the end of the sequence we come back to starting point
            #
            if (self.users_feed_index[user] + self.UNROLL_SIZE) >= self.user_train_max[user]:
                self.users_feed_index[user] = 0

    def create_batch(self):
        self.choose_users()
        self.check_user_is_finished()
        input_value = [self.data[user_number][self.users_feed_index[user_number]:
                                              (self.users_feed_index[user_number])
                                              + self.UNROLL_SIZE] for user_number
                       in self.current_selected_users]
        output_value = [self.data[user_number][self.users_feed_index[user_number] + 1:
                                               (self.users_feed_index[user_number])
                                               + self.UNROLL_SIZE + 1] for user_number
                        in self.current_selected_users]
        output_placeholder = np.array(output_value)
        input_placeholder = np.array(input_value)
        self.move_forward_current_users_index()
        user_numbers = input_placeholder[:, :, 0]
        input_gap_batch = input_placeholder[:, :, 1]
        input_session_size_batch = input_placeholder[:, :, 2]
        output_gap_batch = output_placeholder[:, :, 1]
        output_session_size_batch = output_placeholder[:, :, 2]
        output_session_size_batch = np.expand_dims(output_session_size_batch, -1)
        input_session_size_batch = np.expand_dims(input_session_size_batch, -1)

        #  print(input_gap.shape, output_gap.shape, output_session_length,
        #  input_session_size_batch)
        return input_gap_batch, input_session_size_batch, user_numbers, output_gap_batch, output_session_size_batch

    def set_states(self, user_states):
        self.user_states = user_states

    def get_states(self):
        return self.user_states


class ExtendedBatchFeeder(BatchFeeder):
    def set_n(self):
        pass

    def create_batch(self):
        self.choose_users()
        self.check_user_is_finished()
        input_value = [self.data[user_number][self.users_feed_index[user_number]:
                                              (self.users_feed_index[user_number])
                                              + self.UNROLL_SIZE]
                       for user_number in self.current_selected_users]
        output_value = [self.data[user_number][self.users_feed_index[user_number] + 1:
                                               (self.users_feed_index[user_number])
                                               + self.UNROLL_SIZE + 1] for user_number
                        in self.current_selected_users]
        output_placeholder = np.array(output_value)
        input_placeholder = np.array(input_value)
        self.move_forward_current_users_index()
        user_numbers = input_placeholder[:, :, 0]
        input_gap_batch = input_placeholder[:, :, 1]
        input_session_size_batch = input_placeholder[:, :, 2]
        output_gap_batch = output_placeholder[:, :, 1]
        output_session_size_batch = output_placeholder[:, :, 2]

        input_session_exact_day = input_placeholder[:, :, 3]
        input_session_exact_hour = input_placeholder[:, :, 4]

        # output_session_exact_time = output_placeholder[:, :, 3]
        output_session_size_batch = (np.expand_dims(output_session_size_batch, -1))
        input_session_size_batch = np.expand_dims(input_session_size_batch, -1)
        # print(input_gap.shape, output_gap.shape, output_session_length,
        #       input_session_size_batch)
        return input_gap_batch, input_session_size_batch, user_numbers, \
               output_gap_batch, output_session_size_batch, \
               input_session_exact_day, input_session_exact_hour


class SameBatchFeeder(BatchFeeder):
    def __init__(self):
        super().__init__()

    def choose_users(self):
        self.current_selected_users = np.arange(100)[0:self.batch_size]


class SameValueOfEveryBatch(BatchFeeder):
    def __init__(self):
        super().__init__()

    def choose_users(self):
        self.current_selected_users = np.ones(100)[0:self.batch_size]


class FakeDataGenerator:
    def __init__(self):
        self.data_file_name = "fake_data"
        self.data = self.calculate_fake_data()
        self.address = self.data_address()
        self.calculate_fake_data()
        self.save_fake()
        self.load_fake()

    def data_address(self):
        """""

        """
        return directory_address + self.data_file_name

    def calculate_fake_data(self, length_of_each_data=1000, number_of_users=990,
                            max_exp=10, max_exp_value=10):
        """
        :param length_of_each_data:
        :param number_of_users:
        :param max_exp:
        :param max_exp_value:
        :return  output is with the shape of [u1,u2,u3,ui,...,u1000] if 1000 is number of users:
                where ui is with the shape of [number of users * 3],
                where [i,0] is user index, [i,1] is user session, [i,2] is user
        """
        x = np.linspace(0, max_exp, length_of_each_data)
        value = np.exp(x)
        output_values = []
        for user in range(number_of_users):
            user_session_matrix = np.zeros([1000, 3])
            user_number_matrix = user * np.ones([1000])
            user_session_matrix[:, 0] = user_number_matrix
            user_session_matrix[:, 1] = value
            user_session_matrix[:, 2] = value
            output_values.append(user_session_matrix)
        return output_values

    def load_fake(self):
        """ load wrapped_panda_data """
        with open(self.address, "rb") as file:
            data = pickle.load(file)
        self.data = data
        print(self.data[0][0:4, 0:3])

    def outer_load(self):
        """load wrapped_panda_data"""
        with open(self.address, "rb") as file:
            data = pickle.load(file)
        return data

    def save_fake(self):
        with open(self.address, "wb") as file:
            pickle.dump(self.data, file)


class FakeFeeder(BatchFeeder):

    def load_data(self, address="fake_data"):
        """ load wrapped_panda_data """
        with open(address, "rb") as file:
            data = pickle.load(file)
        self.data = data
        # wrapped_panda_data shape :
        self.user_number = len(data)

    if __name__ == "__main__":
        extractInfo = ExtendInfo()
        # dataAnalysis = DataAnalysis()
        # dataAnalysis.load_data()
        # dataAnalysis.visualize_session_length_histogram()
        # dataAnalysis.visualize_gap_visualization()


class BatchFeeder3D(ExtendedBatchFeeder):
    def __init__(self):
        super().__init__()

    def check_user_is_finished(self):
        for user in self.current_selected_users:
            if (self.users_feed_index[user] + self.UNROLL_SIZE * self.gap_vector_size) >= self.user_train_max[user]:
                self.users_feed_index[user] = 0


    def move_forward_current_users_index(self):
        for user in set(self.current_selected_users):
            print("", set(self.current_selected_users))
            self.users_feed_index[user] = self.users_feed_index[user] + self.step_in_time



    def create_batch(self):
        self.check_user_is_finished()
        self.choose_users()
        input_values = [[self.data[user_number][k + self.users_feed_index[user_number]:
                                                k + self.users_feed_index[user_number] +
                                                self.gap_vector_size]
                         for k in range(self.UNROLL_SIZE)] for user_number in self.current_selected_users]

        output_values = [[self.data[user_number][k + self.users_feed_index[user_number] +
                                                 self.gap_vector_size:
                                                 k + self.users_feed_index[user_number] +
                                                 + self.gap_vector_size + 1]
                          for k in range(self.UNROLL_SIZE)] for user_number in self.current_selected_users]
        input_placeholder = np.array(input_values)
        output_placeholder = np.array(output_values)
        self.move_index()
        user_numbers = input_placeholder[:, :, :, 0]
        input_gap_batch = input_placeholder[:, :, :, 1]  # keep [Batch, Unrolled, vector_size,get gap_size]

        input_session_size_batch = input_placeholder[:, :, :, 2]
        output_gap_batch = output_placeholder[:, :, :, 1]  # keep [Batch. Unrolled_size,vector_Size, get gap_Size]
        # in this case vector_Size is 1

        output_session_size_batch = output_placeholder[:, :, :, 2]
        input_session_exact_day = input_placeholder[:, :, :, 3]
        input_session_exact_hour = input_placeholder[:, :, :, 4]
        return input_gap_batch, input_session_size_batch, user_numbers, \
               output_gap_batch, output_session_size_batch, \
               input_session_exact_day, input_session_exact_hour

    def create_user_test(self):
        for user_number in range(self.dataSize):
            pass

    def move_index(self):
        pass


class TestBatchFeeder(BatchFeeder3D):

    def __init__(self):
        super().__init__()
        self.current_test_user = 0
        self.testing_is_finished = False

    def choose_users(self):
        self.current_selected_users = [self.current_test_user] * self.batch_size

    def next_batch_is_available(self):
        if self.testing_is_finished:
            return False
        else:
            return True

    def reset_testing(self):
        self.testing_is_finished = False
        self.current_test_user = 0
        for user in range(self.dataSize):
            self.users_feed_index[user] = self.user_train_max[user]

    def print_batch_size(self):
        for user in range(self.dataSize):
            print("user : ", user, "batch size : ", self.user_max[user], "train max : ",
                  self.user_train_max[user], "difference : ", self.user_max[user] - self.user_train_max[user])

    def create_batch(self):
        self.check_user_is_finished()
        self.choose_users()
        print("user number :       ", self.current_test_user,
              "user index to feed :", self.users_feed_index[self.current_test_user],
              "max of current user : ",self.user_max[self.current_test_user])
        input_values = [[self.data[user_number][k + self.users_feed_index[user_number]:
                                                k + self.users_feed_index[user_number] +
                                                self.gap_vector_size]
                         for k in range(self.UNROLL_SIZE)] for user_number in self.current_selected_users]
        output_values = [[self.data[user_number][k + self.users_feed_index[user_number] +
                                                 self.gap_vector_size:
                                                 k + self.users_feed_index[user_number] +
                                                 + self.gap_vector_size + 1]
                          for k in range(self.UNROLL_SIZE)] for user_number in self.current_selected_users]
        input_placeholder = np.array(input_values)
        output_placeholder = np.array(output_values)
        self.move_forward_current_users_index()

        user_numbers = input_placeholder[:, :, :, 0]
        input_gap_batch = input_placeholder[:, :, :, 1]  # keep [Batch, Unrolled, vector_size,get gap_size]

        input_session_size_batch = input_placeholder[:, :, :, 2]
        output_gap_batch = output_placeholder[:, :, :, 1]  # keep [Batch. Unrolled_size,vector_Size, get gap_Size]
        # in this case vector_Size is 1

        output_session_size_batch = output_placeholder[:, :, :, 2]
        input_session_exact_day = input_placeholder[:, :, :, 3]
        input_session_exact_hour = input_placeholder[:, :, :, 4]
        return input_gap_batch, input_session_size_batch, user_numbers, \
               output_gap_batch, output_session_size_batch, \
               input_session_exact_day, input_session_exact_hour

    def check_user_is_finished(self):
        if self.users_feed_index[self.current_test_user] + self.UNROLL_SIZE \
                + self.gap_vector_size + 1 >= self.user_max[self.current_test_user]:
            value = self.increment_user_id_test()
            if value == 1:
                print("testing is finished !")
                self.reset_testing()
                self.testing_is_finished = True

    def increment_user_id_test(self):
        if self.current_test_user == 20:
            return 1
        else:
            self.current_test_user += 1
            return 0

    def increment_user_id(self):
        if self.current_test_user == self.dataSize:
            return 1
        else:
            self.current_test_user += 1
            return 0
