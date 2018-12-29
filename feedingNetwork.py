import numpy as np
import pickle
from tensorboardX import SummaryWriter

data_address_pushe = "/home/shayan/simpleRNN/shayanPrep"
data_address_mll = "/home/shayan/shayancode/shayanPrep.pk"
directory_address = "/home/shayan/shayancode/"


class ExtractInfo:
    """ extract user, and number of items in each session  from data"""
    """ """

    def __init__(self):
        """init  values"""
        self.main_data = None
        self.max_user_number = 1000
        self.session_type_number = 1
        # 1 for the gap
        self.input_dimension = self.session_type_number + 1
        self.data_address = data_address_mll

    def load_data(self):
        """save measures"""
        print("start loading da `ta")
        with open(self.data_address, "rb") as data:
            self.main_data = pickle.load(data)

        print("data has been loaded.")

    def calculate_measures(self):
        """calculate data 2 feed to the network"""
        print(len(self.main_data))
        for i in range(1001):
            if self.main_data[i] == []:
                print(i)
        user_session = []
        counter = 0
        for user_index, user_data in self.main_data.items():
            user_session_matrix = np.zeros([len(user_data), self.input_dimension + 1])
            for index, session_data in enumerate(user_data):
                # session data[0] is gap size, session_data[3] is item's list
                # TODO  for multiple session this code should be changed
                user_session_matrix[index, 0] = user_index
                user_session_matrix[index, 1] = session_data[0]
                user_session_matrix[index, 2] = len(session_data[3])
            if user_session_matrix.shape[0] > 30:
                counter = counter + 1
                print(user_session_matrix.shape[0])
                user_session.append(user_session_matrix)

            # print(user_session[counter][1:5, 0:3])
        return user_session

    @staticmethod
    def save_values(data):
        with open("feeding_values", "wb") as file:
            pickle.dump(data, file)

    def extract_and_save(self):
        self.load_data()
        user_sessions = self.calculate_measures()
        self.save_values(user_sessions)


class DataAnalysis(ExtractInfo):

    def load_data(self):
        """save measures"""
        print("start loading data")
        with open(self.data_address, "rb") as data:
            self.main_data = pickle.load(data)
            # print(len(self.main_data)) ->    992 -> number of users
            # print(len(self.main_data[1])) ->    907 -> number of session for first user
            # print(len(self.main_data[1][1])) -> [gap_size(in minutes), 'Day of week', "hour of the day", session Tracks]
            # session Track"
            print(self.main_data[1][1])
            print(type(self.main_data))
            print(self.main_data[1][1])
        print("data has been loaded.")

    def visualize_histogram(self):
        pass


class BatchFeeder:
    """Feeding batch to network"""

    def __init__(self):
        self.UNROLL_SIZE = 16
        self.data = []
        self.user_number = 0
        self.load_data()
        self.user_current_number = np.zeros([1000], int)
        # user current number of index of start of the time
        self.user_max = np.zeros([1000], int)
        # maximum valid time index for data for each user
        self.user_train_max = np.zeros([1000], int)
        #
        self.train_test_split_rate = 0.8
        # if train_test_split = 0.8 : end 20% of data used for test
        # and 80% of first part of data is used for training

        self.batch_size = 64
        # batch_size of the network
        self.dataSize = len(self.data)
        self.current_selected_users = []
        self.step_in_time = 16
        self.HIDDDEN_LSTM_SIZE = 80
        self.user_states = (np.zeros([self.batch_size, self.HIDDDEN_LSTM_SIZE]),
                            np.zeros([self.batch_size, self.HIDDDEN_LSTM_SIZE]))

    def load_data(self, address="feeding_values"):
        """ load data """
        with open(address, "rb") as file:
            data = pickle.load(file)
        self.data = data
        # data shape :
        self.user_number = len(data)

    def set_user_max_train_max_value(self):
        """ set matrix of max_value and train_max_index for each user """
        for user_index, user_sessions in enumerate(self.data):
            self.user_max[user_index] = int(user_sessions.shape[0])
            self.user_train_max[user_index] = int(user_sessions.shape[0] * .8)
            # print(self.user_max[1:10])
            # print(self.user_train_max[1:10])

    def choose_users(self):
        self.current_selected_users = np.random.randint(0, self.dataSize, self.batch_size)
        # print(self.current_selected_users)

    def update_value(self):
        for user in self.current_selected_users:
            self.user_current_number[user] = self.user_current_number[user] + self.step_in_time

    def check_index_possibility(self):
        for user in self.current_selected_users:
            # if we reach the end of the sequence we come back to starting point
            #
            if (self.user_current_number[user] + self.UNROLL_SIZE) >= self.user_max[user]:
                self.user_current_number[user] = 0

    def create_batch(self):
        self.choose_users()
        self.check_index_possibility()
        input_value = [self.data[user_number][self.user_current_number[user_number]:
                                              (self.user_current_number[user_number])
                                              + self.UNROLL_SIZE] for user_number
                       in self.current_selected_users]
        output_value = [self.data[user_number][self.user_current_number[user_number] + 1:
                                               (self.user_current_number[user_number])
                                               + self.UNROLL_SIZE + 1] for user_number
                        in self.current_selected_users]
        output_placeholder = np.array(output_value)
        input_placeholder = np.array(input_value)
        self.update_value()
        user_numbers = input_placeholder[:, :, 0]
        input_gap_batch = input_placeholder[:, :, 1]
        input_session_size_batch = input_placeholder[:, :, 2]
        output_gap_batch = output_placeholder[:, :, 1]
        output_session_size_batch = output_placeholder[:, :, 2]
        output_session_size_batch = (np.expand_dims(output_session_size_batch, -1))
        input_session_size_batch = np.expand_dims(input_session_size_batch, -1)

        # print(input_gap.shape, output_gap.shape, output_session_length,
        #       input_session_size_batch)
        return input_gap_batch, input_session_size_batch, user_numbers, output_gap_batch, output_session_size_batch

    def set_states(self, user_states):
        self.user_states = user_states

    def get_states(self):
        return self.user_states

    # extractInfo = ExtractInfo()


# extractInfo.extract_and_save()
class SameBatchFeeder(BatchFeeder):
    def __init__(self):
        super(SameBatchFeeder, self).__init__()

    def choose_users(self):
        self.current_selected_users = np.arange(100)[0:self.batch_size]


class SameValueOfEveryBatch(BatchFeeder):
    def __init__(self):
        super(SameValueOfEveryBatch, self).__init__()

    def choose_users(self):
        self.current_selected_users = np.ones(100)[0:self.batch_size]


class FakeDataGenerator:
    data_file_name = "fake_data"
    address = directory_address + data_file_name

    @staticmethod
    def data_address():
        """""

        """
        return directory_address + FakeDataGenerator.data_file_name

    def __init__(self):
        self.data = self.calculate_fake_data()
        self.address = self.data_address()
        self.calculate_fake_data()
        self.save_fake()
        self.load_fake()

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
        """ load data """
        with open(self.address, "rb") as file:
            data = pickle.load(file)
        self.data = data
        print(self.data[0][0:4, 0:3])

    @staticmethod
    def outer_load():
        """load data"""
        with open(FakeDataGenerator.address, "rb") as file:
            data = pickle.load(file)
        return data

    def save_fake(self):
        with open(self.address, "wb") as file:
            pickle.dump(self.data, file)


class FakeFeeder(BatchFeeder):

    def load_data(self, address=FakeDataGenerator.data_file_name):
        """ load data """
        with open(address, "rb") as file:
            data = pickle.load(file)
        self.data = data
        # data shape :
        self.user_number = len(data)


if __name__ == "__main__":
    data_analysis = DataAnalysis()
    data_analysis.load_data()
# def create_batch(self):
#     input_gap, input_session_size_batch, user_numbers, output_gap, \
#     output_session_length = super.create_batch(self)
#


#
# batch_feeder = SameBatchFeeder()
# batch_feeder.set_user_max_train_max_value()
# x1, x2, x3, x4, x5 = batch_feeder.create_batch()
# print(x1[0])
# print(x4[0])
# print(x2[0])
# x1, x2, x3, x4, x5 = batch_feeder.create_batch()
# print(x1[0])
# print(x4[0])
#
