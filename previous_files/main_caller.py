from feeders import ExtendedBatchFeeder
from previous_files.Rnn import SimpleRNNWithTruncatedLoss
import tensorflow as tf

# setting batch feeder ###
# Batch Feeder is used to feed wrapped_panda_data to network,

batch_feeder = ExtendedBatchFeeder()
batch_feeder.set_user_max_train_max_value()
print("********-----------------------********")
batch_feeder.create_batch()
print("********-----------------------********")
batch_feeder.create_batch()
batch_feeder.check_user_data()
# setting rnn
rnn = SimpleRNNWithTruncatedLoss()
#
rnn_input = rnn.placeholders2rnn()
# access to placeholders
input_users, retrieved_user_states, input_genre_count, \
input_time, user_lstm_state_reference, output_real_time, \
output_real_counter = rnn.access_place_holders()

lstm_output_value, lstm_states, state, sum_measure, lstm_state_measure = rnn.set_rnn(rnn_input)
time_prediction, number_of_each_genre_prediction, pm1, pm2, g1, g2 = rnn.time_predictor(lstm_output_value)
optimization_step, sumOfLoss, real_loss_1, real_loss_2 = rnn.set_cost(time_prediction, number_of_each_genre_prediction)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# Some variables to have in the scope.
batch_size = 64


# input_gap = np.zeros([batch_size, unroll_size])
# user_numbers = np.ones([batch_size, unroll_size])
# output_gap = np.zeros([batch_size, unroll_size])
# input_session_length = np.zeros([batch_size, unroll_size, 1])
# output_session_length = np.zeros([batch_size, unroll_size, 1])


def config_tensorboard(session):
    """make a writer"""
    log_directory = "/home/shayan/shayancode/tensorboard_directory"
    writer = tf.summary.FileWriter(log_directory)
    writer.add_graph(session.graph)
    return writer


writer = config_tensorboard(sess)
merged_summary = tf.summary.merge_all()
# make a writer #

for i in range(100000):
    input_gap, input_session_length, user_numbers, output_gap, \
    output_session_length, input_session_exact_day, \
    input_session_exact_hour = batch_feeder.create_batch()
    state2Feed = batch_feeder.get_states()
    session_out = sess.run(
        [optimization_step, sumOfLoss, lstm_states, number_of_each_genre_prediction, time_prediction],
        feed_dict={rnn.input_time: input_gap,
                   rnn.input_users: user_numbers,
                   input_genre_count: input_session_length,
                   output_real_counter: output_session_length,
                   output_real_time: output_gap,
                   state[0]: state2Feed[0],
                   state[1]: state2Feed[1],
                   rnn.input_exact_time_day: input_session_exact_day,
                   rnn.input_exact_time_hour: input_session_exact_hour,
                   })
    last_lstm_state = session_out[2]
    batch_feeder.set_states(last_lstm_state)
    # analysis : set state is called in every step of program
    if i % 100 == 0:
        s = sess.run([merged_summary, pm1, pm2, g1, g2, real_loss_1, real_loss_2],
                     feed_dict={input_time: input_gap,
                                input_users: user_numbers,
                                input_genre_count: input_session_length,
                                output_real_counter: output_session_length,
                                output_real_time: output_gap,
                                state[0]: state2Feed[0],
                                state[1]: state2Feed[1],
                                rnn.input_exact_time_day: input_session_exact_day,
                                rnn.input_exact_time_hour: input_session_exact_hour,
                                })

        writer.add_summary(s[0], i)
        print("iteration : ", i)
        print("loss mean : ", session_out[1])
        print("MAE 1 : ", s[5], "MAE 2 : ", s[6])
        print("estimated gap: ", s[1], "estimated session counter", s[2])
        print("real gap: ", s[3], "real session counter", s[4])
        # print("loss1 :", loss[1])
