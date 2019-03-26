from feeding_network import BatchFeeder3D, TestBatchFeederfrom better_rnn import RNNModelimport tensorflow as tfimport numpy as npbatch_feeder = BatchFeeder3D()batch_feeder.set_user_max_train_max_value()# setting rnnrnn = RNNModel()test_feeder = TestBatchFeeder()test_feeder.set_user_max_train_max_value()## some configurationdef config_tensorboard(session):    """make a writer"""    log_directory = "/home/shayan/shayancode/tensorboard_directory"    writer = tf.summary.FileWriter(log_directory)    writer.add_graph(session.graph)    return writerdef config_to_cpu():    config = tf.ConfigProto(device_count={'GPU': 0})    session = tf.Session(config=config)    return sessionsess = tf.InteractiveSession()# sess = config_to_cpu()sess.run(tf.global_variables_initializer())sess.run(tf.local_variables_initializer())batch_size = 64writer = config_tensorboard(sess)merged_summary = tf.summary.merge_all()for i in range(100000):    input_gap, input_session_length, user_numbers, output_gap, \    output_session_length, input_session_exact_day, \    input_session_exact_hour = batch_feeder.create_batch()    state2Feed = batch_feeder.get_states()    session_out = sess.run(        [rnn.optimization_step, rnn.sumOfLoss, rnn.lstm_states, rnn.number_of_each_genre_prediction, rnn.time_prediction],        feed_dict={rnn.input_time: input_gap,                   rnn.user_number: user_numbers,                   rnn.input_genre_count: input_session_length,                   rnn.output_real_counter: output_session_length,                   rnn.output_real_time: output_gap,                   rnn.state[0]: state2Feed[0],                   rnn.state[1]: state2Feed[1],                   rnn.input_exact_time_day: input_session_exact_day,                   rnn.input_exact_time_hour: input_session_exact_hour,                   })    last_lstm_state = session_out[2]    batch_feeder.set_states(last_lstm_state)    # analysis : set state is called in every step of program    if i % 100 == 0:        input_gap, input_session_length, user_numbers, output_gap, \        output_session_length, input_session_exact_day, \        input_session_exact_hour = batch_feeder.create_batch()        s = sess.run([merged_summary, rnn.pm1, rnn.pm2, rnn.g1, rnn.g2, rnn.real_loss_1, rnn.real_loss_2],                     feed_dict={rnn.input_time: input_gap,                                rnn.user_number: user_numbers,                                rnn.input_genre_count: input_session_length,                                rnn.output_real_counter: output_session_length,                                rnn.output_real_time: output_gap,                                rnn.state[0]: state2Feed[0],                                rnn.state[1]: state2Feed[1],                                rnn.input_exact_time_day: input_session_exact_day,                                rnn.input_exact_time_hour: input_session_exact_hour,                                }                     )        writer.add_summary(s[0], i)        print("iteration ", i)        print("loss mean : ", session_out[1])        print("MAE 1", s[5], "MAE 2", s[6])        print("estimated gap: ", s[1], "estimated session counter", s[2])        print("real gap: ", s[3], "real session counter", s[4])        # print("loss1 :", loss[1])    if i % 1000 == 900:        test_feeder.reset_testing()        MAE1 = []        MAE2 = []        while test_feeder.next_batch_is_available() == True:            input_gap, input_session_length, user_numbers, output_gap, \            output_session_length, input_session_exact_day, \            input_session_exact_hour = test_feeder.create_batch()            s = sess.run([merged_summary, rnn.pm1, rnn.pm2, rnn.g1, rnn.g2, rnn.real_loss_1, rnn.real_loss_2],                         feed_dict={rnn.input_time: input_gap,                                    rnn.user_number: user_numbers,                                    rnn.input_genre_count: input_session_length,                                    rnn.output_real_counter: output_session_length,                                    rnn.output_real_time: output_gap,                                    rnn.state[0]: state2Feed[0],                                    rnn.state[1]: state2Feed[1],                                    rnn.input_exact_time_day: input_session_exact_day,                                    rnn.input_exact_time_hour: input_session_exact_hour,                                    })            MAE1.append(s[5])            MAE2.append(s[6])        print("iteration ", i)        print("MAE 1", np.array(MAE1).mean(),              "MAE 2", np.array(MAE2).mean())class dataLink():    pass