import tensorflow as tfimport numpyimport unittestNEURON_NUMBERS = 80  #USER_NUMBERS = 1100  #BATCH_SIZE = 64UNROLLED_SIZE = 64USER_VECTOR_SIZE = 10SEED = 1234HOUR_NUMBER = 24DAY_NUMBER = 7HOUR_IN_WEEK_NUMBER = HOUR_NUMBER * DAY_NUMBERINPUT_VECTOR_SIZE = 16NUMBER_OF_EACH_GENRE = 1OUTPUT_VECTOR_SIZE = 1OUTPUT_VECTOR_SIZE_GENRE = 1USER_EMBEDDING_SIZE = 30EPSILON = tf.constant(10 ** (-8))class RNNModel:    def __init__(self):        # set network's target        self.time_target, self.time_target_log = None, None        self.session_length_target, self.session_length_log = None, None        self.set_target()        # set user embedding        self.users_reference, self.users_indices, self.retrieved_user_states = None, None, None        self.fetch_users_vector()        # set network's input        self.input_session_length, self.input_time = None, None        self.log_input_time, self.input_session_length_log = None, None        self.set_input()        # absolute time        self.exact_hour, self.exact_day, self.exact_hour_in_week = None, None, None        self.set_absoloute_time()        # rnn function        self.run_functions()    # setting input to network    def set_input(self):        try:            with tf.variable_scope("inputs"):                self.input_session_length = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, INPUT_VECTOR_SIZE), name="input_session_length")                self.input_time = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, INPUT_VECTOR_SIZE), name="input_time")                self.__set_input_log()        except Exception as e:            print(str(e))            raise Exception("error occured in setting input values")    def __set_input_log(self):        self.log_input_time = tf.log(self.input_time)        self.input_session_length_log = tf.log(self.input_session_length)    # user embedding    def fetch_users_vector(self):        try:            with tf.variable_scope("user_embedding"):                self.users_reference = tf.Variable(tf.random_uniform([USER_NUMBERS, USER_EMBEDDING_SIZE], -1.0, 1.0),                                                   trainable=True, name="user_embedding_matrix")                self.users_indices = tf.placeholder(tf.int32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="user_number")                self.retrieved_user_states = tf.nn.embedding_lookup(self.users_reference, self.users_indices, name="user_states")                return 0        except Exception as e:            print(str(e))            raise Exception("error in accesing user reference")    # target functions    def set_target(self):        try:            with tf.variable_scope("target_values"):                self.time_target = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, OUTPUT_VECTOR_SIZE),                                                  name="time_target")                self.session_length_target = tf.placeholder(tf.float32, shape=(BATCH_SIZE, UNROLLED_SIZE, OUTPUT_VECTOR_SIZE_GENRE),                                                            name="session_length_target")                self.__set_target_log()        except Exception as e:            print(str(e))            raise Exception("Error occured while setting the target!!")    def __set_target_log(self):        try:            with tf.variable_scope("target_log"):                self.time_target_log = tf.log(self.time_target)                self.session_length_log = tf.log(self.session_length_target)        except Exception as e:            print(str(e))            raise Exception("error occured inside settin target log")    def set_absoloute_time(self):        try:            with tf.variable_scope("absolute_time"):                self.exact_hour = tf.placeholder(tf.int32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="absolute_hour")                self.exact_day = tf.placeholder(tf.int32, shape=(BATCH_SIZE, UNROLLED_SIZE), name="absoloute_day")                # self.input_exact_time_in_week = self.exact_day * 7 + self.exact_hour                self.exact_hour_in_week = self.exact_hour * tf.constant(24, tf.int32) + self.exact_day                return 0        except Exception as e:            print(str(e))            raise Exception("an error occured in absolute time setting")    def placeholders_2_rnn(self):        hour_in_week = tf.Variable(tf.random_uniform([HOUR_IN_WEEK_NUMBER, USER_VECTOR_SIZE], -1.0, 1.0),                                   name="time_embedding_verctor")        hour_in_week_embedding = tf.nn.embedding_lookup(hour_in_week, self.exact_hour_in_week,                                                        name="hour_in_week_embedded_vector")        expanded_time_input = self.log_input_time        self.size_of_user_state = 10        self.size_of_hour_state = 10        retrieved_user_state = tf.layers.dense(self.retrieved_user_states, self.size_of_user_state, activation=tf.nn.relu)        hour_of_week = tf.layers.dense(hour_in_week_embedding, self.size_of_hour_state, activation=tf.nn.relu)        # retrieved_user_state = tf.squeeze(retrieved_user_state)        # hour_of_week = tf.squeeze(hour_of_week)        input2rnn = tf.concat([            expanded_time_input, self.input_session_length_log, retrieved_user_state, hour_of_week], 2,            name="rnn_input_unrolled")        print("input2rnn.shape : ", input2rnn.shape)        return input2rnn    def set_rnn(self, lstm_input, name="RNN"):        with tf.variable_scope(name):            basic_cell = tf.nn.rnn_cell.LSTMCell(NEURON_NUMBERS, name="lstm_cell")            self.sequence_length = tf.placeholder(tf.int32, shape=(BATCH_SIZE,))            self.lstm_input_c = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NEURON_NUMBERS))            self.lstm_input_h = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NEURON_NUMBERS))            self.lstm_input_state = tf.nn.rnn_cell.LSTMStateTuple(self.lstm_input_c, self.lstm_input_h)            sum_measure = tf.reduce_sum(self.lstm_input_state)            lstm_output, self.lstm_ouput_state = tf.nn.dynamic_rnn(cell=basic_cell, inputs=lstm_input, sequence_length=self.sequence_length,                                                                   initial_state=self.lstm_input_state)            (self.lstm_ouput_c, self.lstm_ouput_h) = self.lstm_ouput_state            lstm_state_measure = tf.reduce_sum(self.lstm_ouput_state)            return lstm_output, sum_measure, lstm_state_measure    @staticmethod    def normalize_variable(time_prediction, number_prediction, log_output_time, log_output_number):        v1 = EPSILON * tf.ones_like(number_prediction)        v2 = EPSILON * tf.ones_like(time_prediction)        time_normalization = (v2 + log_output_time)        time_normalization_mean = tf.reduce_mean(time_normalization)        session_size_normalization = (v1 + log_output_number)        session_size_normalization_mean = tf.reduce_mean(session_size_normalization)        normalized_output_session_size = tf.div(log_output_number, session_size_normalization_mean)        normalized_output_time = tf.div(log_output_time, time_normalization_mean)        normalized_time_prediction = tf.div(time_prediction, time_normalization_mean)        normalized_session_size = tf.div(number_prediction, session_size_normalization_mean)        return normalized_time_prediction, normalized_output_time, normalized_session_size, \               normalized_output_session_size    def access_place_holders(self):        """make access to variable outside class scope"""        return (self.users_indices, self.retrieved_user_states,                self.input_session_length, self.input_time,                self.users_reference, self.time_target,                self.session_length_target)    def time_predictor(self, lstm_outputs):        t11 = tf.layers.dense(self.log_input_time, 100)        t12 = tf.layers.dense(self.input_session_length_log, 100)        trnn1 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))        t1 = tf.concat([t11, trnn1], axis=-1)        trnn2 = tf.layers.dense(lstm_outputs, 100, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),                                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))        t2 = tf.concat([t12, trnn2], axis=-1)        time_prediction = tf.layers.dense(t1, OUTPUT_VECTOR_SIZE, name="gap_predictor",                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),                                          bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1))        time_prediction_rectified = tf.nn.softplus(time_prediction, "gap_positive")        number_of_each_genre_prediction = tf.layers.dense(t2, OUTPUT_VECTOR_SIZE_GENRE, name="genre_predictor",                                                          kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),                                                          bias_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1)                                                          )        number_of_each_genre_prediction_rectified = tf.nn.softplus(number_of_each_genre_prediction,                                                                   name="genre_positive")        time_prediction_exp = time_prediction_rectified        number_prediction_exp = number_of_each_genre_prediction_rectified        print("number_of_each_genre_prediction", number_of_each_genre_prediction_rectified)        print("time_prediction_exp", time_prediction_exp)        pm1 = tf.reduce_mean(time_prediction_exp)        pm2 = tf.reduce_mean(number_prediction_exp)        real_out1 = tf.reduce_mean(self.time_target_log)        real_out2 = tf.reduce_mean(self.session_length_log)        return time_prediction_exp, number_prediction_exp, pm1, pm2, real_out1, real_out2    def set_cost(self, time_prediction, number_prediction, name="cost", have_regulizer=True):        with tf.name_scope(name):            # l1 = tf.losses.mean_squared_error(tf.expand_dims(self.output_time_real, -1), time_prediction)            # l1 = tf.nn.log_poisson_loss(tf.expand_dims(self.time_target, -1), time_prediction, name="loss1",            #                             compute_full_loss=True)            # l2 = tf.nn.log_poisson_loss(self.session_length_target, number_prediction, name="loss2",            #                             compute_full_loss=True)            tf.summary.histogram("time of real outputs", self.time_target)            tf.summary.histogram("Number of count in a session", self.session_length_target)            tf.summary.histogram("generated time histogram", time_prediction)            tf.summary.histogram("generated count values", number_prediction)            # self.normalize_variable(time_prediction, number_prediction,            #                         self.time_target_log, self.session_length_log)            rl1 = tf.losses.mean_squared_error(                self.time_target_log, time_prediction)            rl2 = tf.losses.mean_squared_error(                number_prediction, self.session_length_log)            mae1 = tf.losses.absolute_difference(tf.div(tf.exp(time_prediction), tf.constant(60, tf.float32)),                                                 tf.div(tf.exp(self.time_target_log),                                                        tf.constant(60, tf.float32)))            mae2 = tf.losses.absolute_difference(tf.div(tf.exp(number_prediction), tf.constant(60, tf.float32)),                                                 tf.div(tf.exp(self.session_length_log),                                                        tf.constant(60, tf.float32)))            tf.summary.scalar("output real time mean : ", tf.reduce_mean(self.time_target_log))            tf.summary.scalar("output real season size : ", tf.reduce_mean(self.session_length_log))            # loss = tf.add(tf.reduce_mean(l1), tf.reduce_mean(l2))            loss_without_regulizer = tf.add(rl1, rl2)            # loss = self.apply_variable_regulizer(loss_without_regulizer, have_regulizer)            # augmented_loss = self.apply_all_regulizer(loss)            augmented_loss = loss_without_regulizer            optimization_step = tf.train.AdamOptimizer(1e-3).minimize(augmented_loss, name="optimizer2Optimize")            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)            # self.cost = optimizer.minimize(loss=loss)            tf.summary.scalar("loss", augmented_loss)        return optimization_step, augmented_loss, mae1, mae2    def apply_variable_regulizer(self, loss_without_reg, with_regulizer):        if with_regulizer:            number_of_itmes = USER_NUMBERS * USER_EMBEDDING_SIZE            regulizer_coeff = 10 ** (-2)            reg_sum = number_of_itmes / regulizer_coeff            loss = loss_without_reg + tf.div(tf.nn.l2_loss(self.users_reference), reg_sum)        else:            loss = loss_without_reg        return loss    def apply_all_regulizer(self, loss):        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)        augmented_loss = loss + sum(reg_losses)        return augmented_loss    def run_functions(self):        rnn_input = self.placeholders_2_rnn()        self.users_indices, self.retrieved_user_states, self.input_session_length, \        self.input_time, self.users_reference, self.time_target, \        self.output_real_counter = self.access_place_holders()        self.lstm_output_value, self.sum_measure, self.lstm_state_measure = self.set_rnn(rnn_input)        self.time_prediction, self.number_of_each_genre_prediction, self.pm1, self.pm2, self.g1, self.g2 = self.time_predictor(self.lstm_output_value)        self.optimization_step, self.sumOfLoss, self.real_loss_1, self.real_loss_2 = self.set_cost(self.time_prediction, self.number_of_each_genre_prediction)def calculate_size(tensor):    tensor_shape = tf.shape(tensor)    size = 1    for dimension_size in tensor_shape:        size *= dimension_size    return size