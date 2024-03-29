import tensorflow as tf
import numpy

NEURON_NUMBERS = 80  #
USER_NUMBERS = 1100  #
BATCH_SIZE = 64
UNROLLED_SIZE = 16
USER_VECTOR_SIZE = 10
SEED = 1234
HOUR_NUMBER = 24
DAY_NUMBER = 7
HOUR_IN_WEEK_Number = HOUR_NUMBER * DAY_NUMBER
INPUT_VECTOR_SIZE = 16
NUMBER_OF_EACH_GENRE = 1
OUTPUT_VECTOR_SIZE = 1
OUTPUT_VECTOR_SIZE_GENRE = 1


class RNNModel:
    def __init__(self):
        self.output_real_time = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, OUTPUT_VECTOR_SIZE),
                                               name="output_time_real")
        self.output_real_session_size = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, OUTPUT_VECTOR_SIZE_GENRE),
                                                       name="ogg")
        self.user_lstm_state_reference = tf.Variable(tf.random_uniform([USER_NUMBERS, 100], -1.0, 1.0),
                                                     trainable=True)
        self.input_users = tf.placeholder(tf.int32, shape=(BATCH_SIZE, UNROLLED_SIZE, INPUT_VECTOR_SIZE), name="iu")
        self.retrieved_user_states = tf.nn.embedding_lookup(self.user_lstm_state_reference, self.input_users,
                                                            name="user_states")
        self.input_genre_count = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, INPUT_VECTOR_SIZE), name="ig")
        self.input_time = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, INPUT_VECTOR_SIZE), name="it")
        self.log_input_time = tf.log(self.input_time)
        self.log_output_time = tf.log(self.output_real_time)
        self.log_input_session_length = tf.log(self.input_genre_count)
        self.log_output_session_length = tf.log(self.output_real_session_size)
        self.input_exact_time_hour = tf.placeholder(tf.int32, shape=(BATCH_SIZE, UNROLLED_SIZE, INPUT_VECTOR_SIZE))
        self.input_exact_time_day = tf.placeholder(tf.int32, shape=(BATCH_SIZE, UNROLLED_SIZE, INPUT_VECTOR_SIZE))
        # self.input_exact_time_in_week = self.exact_day * 7 + self.exact_hour
        # TODO next line should be changed @RNN TOO importanttttttt!!!!!!!!!
        self.exact_hour_in_week = self.input_exact_time_hour * tf.constant(24, tf.int32) + self.input_exact_time_day

        self.epsilon = tf.constant(10 ** (-8))

    def placeholders2rnn(self):
        user_embedding_reference = tf.Variable(tf.random_uniform([USER_NUMBERS, USER_VECTOR_SIZE], -1.0, 1.0),
                                               name="user_vectors")
        hour_in_week = tf.Variable(tf.random_uniform([HOUR_IN_WEEK_Number, USER_VECTOR_SIZE], -1.0, 1.0),
                                   name="time_embedding_verctor")
        user_embedding_batch = tf.nn.embedding_lookup(user_embedding_reference, self.input_users,
                                                      name="user_embedding_batch")
        hour_in_week_embedding_batch = tf.nn.embedding_lookup(hour_in_week, self.exact_hour_in_week,
                                                              name="week_times")
        expanded_time_input = self.log_input_time

        retrieved_user_state = tf.layers.dense(self.retrieved_user_states, 1, activation=tf.nn.relu)
        hour_of_week = tf.layers.dense(hour_in_week_embedding_batch, 1, activation=tf.nn.relu)
        retrieved_user_state = tf.squeeze(retrieved_user_state)
        hour_of_week = tf.squeeze(hour_of_week)
        print(expanded_time_input.shape, user_embedding_batch.shape, self.input_genre_count.shape)

        input2rnn = tf.concat([
            expanded_time_input, self.log_input_session_length, retrieved_user_state
            , hour_of_week], 2,
            name="rnn_input_unrolled")
        print(input2rnn.shape)
        return input2rnn

    def set_rnn(self, lstm_input, name="RNN"):
        with tf.variable_scope(name):
            basic_cell = tf.nn.rnn_cell.LSTMCell(NEURON_NUMBERS, name="lstm_cell")
            v1 = basic_cell.zero_state(BATCH_SIZE, tf.float32)
            state = tf.Variable(v1, trainable=False)
            sum_measure = tf.reduce_sum(state)
            lstm_output, lstm_state = tf.nn.dynamic_rnn(cell=basic_cell, inputs=lstm_input, initial_state=v1)
            lstm_state_measure = tf.reduce_sum(lstm_state)
            return lstm_output, lstm_state, state, sum_measure, lstm_state_measure

    def normalize_variable(self, time_prediction, number_prediction, log_output_time, log_output_number):
        v1 = self.epsilon * tf.ones_like(number_prediction)
        v2 = self.epsilon * tf.ones_like(time_prediction)
        time_normalization = (v2 + log_output_time)
        time_normalization_mean = tf.reduce_mean(time_normalization)
        session_size_normalization = (v1 + log_output_number)
        session_size_normalization_mean = tf.reduce_mean(session_size_normalization)

        normalized_output_session_size = tf.div(log_output_number, session_size_normalization_mean)
        normalized_output_time = tf.div(log_output_time, time_normalization_mean)
        normalized_time_prediction = tf.div(time_prediction, time_normalization_mean)
        normalized_session_size = tf.div(number_prediction, session_size_normalization_mean)

        return normalized_time_prediction, normalized_output_time, normalized_session_size, \
               normalized_output_session_size

    def access_place_holders(self):
        """make access to variable outside class scope"""
        return (self.input_users, self.retrieved_user_states,
                self.input_genre_count, self.input_time,
                self.user_lstm_state_reference, self.output_real_time,
                self.output_real_session_size)

    def time_predictor(self, lstm_outputs):
        t11 = tf.layers.dense(self.log_input_time, 100)
        t12 = tf.layers.dense(self.log_input_session_length, 100)
        trnn1 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)

        t1 = tf.concat([t11, trnn1], axis=-1)
        trnn2 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)
        t2 = tf.concat([t12, trnn2], axis=-1)
        time_prediction = tf.layers.dense(t1, OUTPUT_VECTOR_SIZE, name="gap_predictor")
        time_prediction_rectified = tf.nn.softplus(time_prediction, "gap_positive")
        number_of_each_genre_prediction = tf.layers.dense(t2, OUTPUT_VECTOR_SIZE_GENRE, name="genre_predictor")
        number_of_each_genre_prediction_rectified = tf.nn.softplus(number_of_each_genre_prediction,
                                                                   name="genre_positive")
        time_prediction_exp = time_prediction_rectified
        number_prediction_exp = number_of_each_genre_prediction_rectified
        print("number_of_each_genre_prediction", number_of_each_genre_prediction_rectified)
        print("time_prediction_exp", time_prediction_exp)
        pm1 = tf.reduce_mean(time_prediction_exp)
        pm2 = tf.reduce_mean(number_prediction_exp)
        real_out1 = tf.reduce_mean(self.log_output_time)
        real_out2 = tf.reduce_mean(self.log_output_session_length)
        return time_prediction_exp, number_prediction_exp, pm1, pm2, real_out1, real_out2

    def set_cost(self, time_prediction, number_prediction, name="cost"):
        with tf.name_scope(name):
            # l1 = tf.losses.mean_squared_error(tf.expand_dims(self.output_time_real, -1), time_prediction)
            # l1 = tf.nn.log_poisson_loss(tf.expand_dims(self.time_target, -1), time_prediction, name="loss1",
            #                             compute_full_loss=True)
            # l2 = tf.nn.log_poisson_loss(self.session_length_target, number_prediction, name="loss2",
            #                             compute_full_loss=True)
            tf.summary.histogram("time of real outputs", self.output_real_time)
            tf.summary.histogram("Number of count in a session", self.output_real_session_size)
            tf.summary.histogram("generated time histogram", time_prediction)
            tf.summary.histogram("generated count values", number_prediction)
            self.normalize_variable(time_prediction, number_prediction,
                                        self.log_output_time, self.log_output_session_length)
            rl1 = tf.losses.mean_squared_error(
                self.log_output_time, time_prediction)
            rl2 = tf.losses.mean_squared_error(
                number_prediction, self.log_output_session_length)
            mae1 = tf.losses.absolute_difference(tf.div(tf.exp(time_prediction), tf.constant(60, tf.float32)),
                                                 tf.div(tf.exp(self.log_output_time),
                                                        tf.constant(60, tf.float32)))
            mae2 = tf.losses.absolute_difference(tf.div(tf.exp(number_prediction), tf.constant(60, tf.float32)),
                                                 tf.div(tf.exp(self.log_output_session_length),
                                                        tf.constant(60, tf.float32)))
            tf.summary.scalar("output real time mean : ", tf.reduce_mean(self.log_output_time))
            tf.summary.scalar("output real season size : ", tf.reduce_mean(self.log_output_session_length))
            # loss = tf.add(tf.reduce_mean(l1), tf.reduce_mean(l2))
            loss = tf.add(rl1, rl2)
            optimization_step = tf.train.AdamOptimizer(1e-3).minimize(loss, name="optimizer2Optimize")
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
            # self.cost = optimizer.minimize(loss=loss)
            tf.summary.scalar("loss", loss)
        return optimization_step, loss, mae1, mae2
