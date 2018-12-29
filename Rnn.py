"""
it has been written by AmirShayan Haghipour.

this file contains expected rnn model,

current pep 8 as a standard guide for codding style
https://legacy.python.org/dev/peps/pep-0008/#class-names
https://legacy.python.org/dev/peps/pep-0484/
TODO  check difference
TODO Check licensing -> https://help.github.com/articles/licensing-a-repository/

"""
import tensorflow as tf
import numpy

NEURON_NUMBERS = 80  #
USER_NUMBERS = 1100  # TODO it should be inferred, if you are not sure about the value
BATCH_SIZE = 64
UNROLLED_SIZE = 16
GENRE_NUMBER = 1  # TODO should be set
USER_VECTOR_SIZE = 10

SEED = 1234

tf.set_random_seed(SEED)
numpy.random.seed(SEED)


class RNNModel:
    def __init__(self):
        # self.output_pl = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="output_time_real")
        # self.output_real_time = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="output_time_real")
        self.output_real_time = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="output_time_real")
        self.output_real_session_size = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, GENRE_NUMBER),
                                                       name="ogg")
        # TODO -> random_uniform initializer  (get tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="output_time_real")variable va variable dar )
        self.user_lstm_state_reference = tf.Variable(tf.random_uniform([USER_NUMBERS, NEURON_NUMBERS], -1.0, 1.0),
                                                     trainable=True)
        self.input_users = tf.placeholder(tf.int32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="iu")
        self.retrieved_user_states = tf.nn.embedding_lookup(self.user_lstm_state_reference, self.input_users,
                                                            name="user_states")
        self.input_genre_count = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, GENRE_NUMBER), name="ig")
        self.input_time = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="it")
        self.log_input_time = tf.log(self.input_time)
        self.log_output_time = tf.log(self.output_real_time)
    def accessToPlaceHolders(self):
        return (self.input_users, self.retrieved_user_states,
                self.input_genre_count, self.input_time,
                self.user_lstm_state_reference, self.output_real_time,
                self.output_real_session_size)

    def placeholders2rnn(self):
        # TODO
        user_embedding_reference = tf.Variable(tf.random_uniform([USER_NUMBERS, USER_VECTOR_SIZE], -1.0, 1.0),
                                               name="user_vectors")
        assert (user_embedding_reference.shape == (USER_NUMBERS, USER_VECTOR_SIZE))
        user_embedding_batch = tf.nn.embedding_lookup(user_embedding_reference, self.input_users,
                                                      name="user_embedding_batch")
        # transposed_genre = tf.transpose(input_genre_count, (1, 2, 0), name="fed_genre_count")
        expanded_time_input = tf.expand_dims(self.log_input_time, -1)
        # print(self.input_time)
        # print(expanded_time_input)

        # checking sizes
        print("check sizes :")
        print(expanded_time_input.shape, user_embedding_batch.shape, self.input_genre_count.shape)
        input2rnn = tf.concat([expanded_time_input, self.input_genre_count], 2,
                              name="rnn_input_unrolled")

        # TODO product inputs by some values before feeding to RNN
        return input2rnn

    def set_rnn(self, lstm_input, name="RNN"):
        with tf.variable_scope(name):
            basic_cell = tf.nn.rnn_cell.BasicLSTMCell(NEURON_NUMBERS, name="lstm_cell")
            # state = basic_cell.zero_state(batch_size=BATCH_SIZE, dtype=tf.float32)
            state = tf.Variable(basic_cell.zero_state(BATCH_SIZE, tf.float32), trainable=False)
            c, h = tf.unstack(state, axis=0)
            state2 = tf.nn.rnn_cell.LSTMStateTuple(c, h)
            sum_measure = tf.reduce_sum(state)
            lstm_output, lstm_state = tf.nn.dynamic_rnn(cell=basic_cell,
                                                        inputs=lstm_input,
                                                        initial_state=state2)
            lstm_state_measure = tf.reduce_sum(lstm_state)
            return lstm_output, lstm_state, state, sum_measure, lstm_state_measure

    def time_predictor(self, lstm_outputs):
        # with tf.variable_scope('FullyConnected'):
        #     W = tf.get_variable('Wt', [NEURON_NUMBERS, 1])
        #     b = tf.get_variable('bt', [1], initializer=tf.constant_initializer(0.0))
        # logits_t = tf.matmul(lstm_outputs, W) + b
        t1 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)
        t2 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)
        time_prediction = tf.layers.dense(t1, 1, name="gap_predictor")
        time_prediction_rectified = tf.nn.softplus(time_prediction, "gap_positive")
        number_of_each_genre_prediction = tf.layers.dense(t2, GENRE_NUMBER, name="genre_predictor")
        number_of_each_genre_prediction_rectified = tf.nn.softplus(number_of_each_genre_prediction,
                                                                   name="genre_positive")
        print(number_of_each_genre_prediction_rectified.shape)
        time_prediction_exp = tf.exp(tf.exp(time_prediction_rectified))
        number_prediction_exp = tf.exp(number_of_each_genre_prediction_rectified)
        pm1 = tf.reduce_mean(time_prediction_exp)
        pm2 = tf.reduce_mean(number_prediction_exp)
        print(pm1)
        print(pm2)

        real_out1 = tf.reduce_mean(self.log_output_time)
        real_out2 = tf.reduce_mean(self.output_real_session_size)
        return time_prediction_exp, number_prediction_exp, pm1, pm2, real_out1, real_out2

    def set_cost(self, time_prediction, number_prediction, name="cost"):
        with tf.name_scope(name):
            # l1 = tf.losses.mean_squared_error(tf.expand_dims(self.output_time_real, -1), time_prediction)
            # l1 = tf.nn.log_poisson_loss(tf.expand_dims(self.output_real_time, -1), time_prediction, name="loss1",
            #                             compute_full_loss=True)
            # l2 = tf.nn.log_poisson_loss(self.output_real_session_size, number_prediction, name="loss2",
            #                             compute_full_loss=True)
            tf.summary.histogram("time of real outputs", self.output_real_time)
            tf.summary.histogram("Number of count in session", self.output_real_session_size)
            tf.summary.histogram("generated time histogram", time_prediction)
            tf.summary.histogram("generated count values", number_prediction)
            epsilon = tf.constant(10 ** (-8))
            print("I am still alive")
            v1 = epsilon * tf.ones_like(number_prediction)
            v2 = epsilon * tf.ones_like(time_prediction)
            time_normalization = (v2 + tf.expand_dims(self.log_output_time, -1))
            # time_normalization = (v2 + time_prediction)
            print(time_normalization)
            time_normalization_mean = tf.reduce_mean(time_normalization)
            print(time_normalization_mean)
            session_size_normalization = tf.add(v1, self.output_real_session_size)
            session_size_normalization_mean = tf.reduce_mean(session_size_normalization)
            normalized_output_session_size = tf.div(self.output_real_session_size, session_size_normalization_mean)
            normalized_output_time = tf.div(tf.expand_dims(self.log_output_time, -1), time_normalization_mean)
            normalized_time_prediction = tf.div(time_prediction, time_normalization_mean)
            normalized_session_size = tf.div(number_prediction, session_size_normalization_mean)

            l1 = tf.losses.mean_squared_error(normalized_output_time,
                                              normalized_time_prediction)

            l2 = tf.losses.mean_squared_error(normalized_output_session_size,
                                              normalized_session_size)
            real_loss_squared_1 = tf.losses.mean_squared_error(
                tf.expand_dims(self.log_output_time, -1),
                time_prediction)
            real_loss_squared_2 = tf.losses.mean_squared_error(
                number_prediction, self.output_real_session_size
            )
            real_loss_1 = tf.sqrt(real_loss_squared_1)
            real_loss_2 = tf.sqrt(real_loss_squared_2)

            tf.summary.scalar("output real time mean", tf.reduce_mean(self.log_output_time))
            tf.summary.scalar("output real season size", tf.reduce_mean(self.output_real_session_size))
            loss = tf.add(tf.reduce_mean(l1), tf.reduce_mean(l2))
            optimization_step = tf.train.AdamOptimizer(1e-3).minimize(loss, name="optimizer2Optimize")
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
            # self.cost = optimizer.minimize(loss=loss)
            tf.summary.scalar("loss", loss)
        return optimization_step, loss, real_loss_1, real_loss_2


class SimpleRNNPoissonCost(RNNModel):
    def time_predictor(self, lstm_outputs):
        t11 = tf.layers.dense(tf.expand_dims(self.log_input_time, -1), 100)
        t12 = tf.layers.dense(self.input_genre_count, 100)
        trnn1 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)
        print(t11)
        print(trnn1)
        t1 = tf.concat([t11, trnn1], axis=-1)
        trnn2 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)
        t2 = tf.concat([t12, trnn2], axis=-1)
        time_prediction = tf.layers.dense(t1, 1, name="gap_predictor")
        time_prediction_rectified = tf.nn.softplus(time_prediction, "gap_positive")
        number_of_each_genre_prediction = tf.layers.dense(t2, GENRE_NUMBER, name="genre_predictor")
        number_of_each_genre_prediction_rectified = tf.nn.softplus(number_of_each_genre_prediction,
                                                                   name="genre_positive")
        print(number_of_each_genre_prediction_rectified.shape)
        pm1 = tf.reduce_mean(time_prediction_rectified)
        pm2 = tf.reduce_mean(number_of_each_genre_prediction_rectified)
        real_out1 = tf.reduce_mean(self.log_output_time)
        real_out2 = tf.reduce_mean(self.output_real_session_size)
        return time_prediction_rectified, number_of_each_genre_prediction_rectified, pm1, pm2, real_out1, real_out2

    def set_cost(self, time_prediction, number_prediction, name="cost"):
        l1 = tf.nn.log_poisson_loss(tf.expand_dims(self.log_output_time, -1), time_prediction, name="loss1",
                                    compute_full_loss=True)
        l2 = tf.nn.log_poisson_loss(self.output_real_session_size, number_prediction, name="loss2",
                                    compute_full_loss=True)
        loss = tf.add(tf.reduce_mean(l1), tf.reduce_mean(l2))
        optimization_step = tf.train.AdamOptimizer(3e-3).minimize(loss, name="optimizer2Optimize")
        tf.summary.scalar("loss", loss)
        return optimization_step, loss


class RnnMlp(RNNModel):
    def time_predictor(self, lstm_outputs):
        t11 = tf.layers.dense(tf.expand_dims(self.log_input_time, -1), 100)
        t12 = tf.layers.dense(self.input_genre_count, 100)
        trnn1 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)

        t1 = tf.concat([t11, trnn1], axis=-1)
        trnn2 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu)
        t2 = tf.concat([t12, trnn2], axis=-1)
        time_prediction = tf.layers.dense(t1, 1, name="gap_predictor")
        time_prediction_rectified = tf.nn.softplus(time_prediction, "gap_positive")
        number_of_each_genre_prediction = tf.layers.dense(t2, GENRE_NUMBER, name="genre_predictor")
        number_of_each_genre_prediction_rectified = tf.nn.softplus(number_of_each_genre_prediction,
                                                                   name="genre_positive")
        time_prediction_exp = time_prediction_rectified
        number_prediction_exp = number_of_each_genre_prediction_rectified
        pm1 = tf.reduce_mean(time_prediction_exp)
        pm2 = tf.reduce_mean(number_prediction_exp)
        real_out1 = tf.reduce_mean(self.log_output_time)
        real_out2 = tf.reduce_mean(self.output_real_session_size)
        return time_prediction_exp, number_prediction_exp, pm1, pm2, real_out1, real_out2
