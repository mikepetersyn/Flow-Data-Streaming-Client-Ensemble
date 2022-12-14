import csv
import math
import defaultParser
from feature_selector import feature_selector

NUM_PARALLEL_EXEC_UNITS = 0


class WDNN:
    def __init__(self):
        #---------------------------------------------------------------------------------------------------- ARG PARSER
        self.FLAGS = None
        self.arg_parse()
         #--------------------------------------------------------------------------------------------- FEATURE SELECTOR
        self.feature_selector = feature_selector
        #-------------------------------------------------------------------------------------- INIT DNN INPUT PARAMETER
        self.input_size = 0
        self.num_classes = 0
        self._init_dnn_parameter()
        #----------------------------------------------------------------------------- DEFINE DNN VARIABLES AND BUILD IT
        self.sess = None
        self.x = None
        self.y_ = None
        self.initial_layer_weights = []
        self.keep_prob_input = None
        self.keep_prob_hidden = None
        self.logits = None
        self.train_step = None
        self.correct_prediction_tr = None
        self.accuracy_tr = None
        self.get_prediction = None
        self.get_pred_prob = None

        self.class_weights = None

        self.dataset = None
        self.class_weighting_block = None
        self.build_dnn()
        # self.train_dataset = None
        # self.test_dataset = None
        # self.class_weighting_block_train = None
        # self.class_weighting_block_test = None

    def arg_parse(self):
        """ create command line parser """
        parser = defaultParser.create_default_parser()
        self.FLAGS, _ = parser.parse_known_args()
        defaultParser.printFlags(self.FLAGS)

    def _init_dnn_parameter(self):
        """ evaluate input parameter for the DNN configuration """
        num_features = 0
        for feature_str in self.FLAGS.features:

            feature = feature_str.split('-')
            feature_type_len = self.feature_selector.get(int(feature[0]))
            if not feature_type_len:
                raise Exception('invalid feature selection')
            feature_len = feature_type_len[int(feature[1])]
            if feature_len == -1:
                raise Exception('invalid feature type')
            num_features += feature_len
        self.input_size = num_features
        self.num_classes = len(self.FLAGS.boundaries)

    def feed_dict(self, test=False, shuffle=False):
        ''' feed dict function '''
        if not test:
          xs, ys = self.dataset.next_batch(self.FLAGS.batch_size, shuffle=shuffle)
          k_h = self.FLAGS.dropout_hidden
          k_i = self.FLAGS.dropout_input
        else:
          xs, ys = self.dataset.next_batch(self.FLAGS.batch_size, shuffle=shuffle)
          k_h = 1.0
          k_i = 1.0

        feed_dict_ = {
            self.x: xs,  # data
            self.y_: ys,  # labels
            self.keep_prob_input: k_i,  # dropout probability input layer
            self.keep_prob_hidden: k_h,  # dropout probability hidden layer
            }

        if self.FLAGS.cw_method == 0:  # udpate feed dict with class weights
          if not test:
            feed_dict_[self.class_weights] = self.class_weighting_block
          else:
            feed_dict_[self.class_weights] = self.class_weighting_block
        return feed_dict_

    def reset_weights(self):
        """ re-initialize weights in dnn graph """
        self.sess.run(self.initial_layer_weights)

    def train(self):
        batches_per_epoch_train = math.ceil(self.dataset.images.shape[0] / self.FLAGS.batch_size)
        if self.FLAGS.fixed_num_epochs:
            batches_per_epoch_train = self.FLAGS.epochs_windows * batches_per_epoch_train
        for _ in range(batches_per_epoch_train):
            self.sess.run(self.train_step, feed_dict=self.feed_dict())

    def test(self):
        acc = 0.
        y = []
        y_pred_prob = []
        batches_per_epoch_test = math.ceil(self.dataset.images.shape[0] / self.FLAGS.batch_size)  # how many batches per test epoch
        for _ in range(batches_per_epoch_test):  # test for a full test epoch
            batch_acc, batch_y, batch_y_pred_prob= self.sess.run(fetches=[self.accuracy_tr, self.y_, self.get_pred_prob], feed_dict=self.feed_dict(test=True))
            acc += batch_acc
            y.append(batch_y)
            y_pred_prob.append(batch_y_pred_prob)
        acc /= batches_per_epoch_test
        return acc, y, y_pred_prob

    def build_dnn(self):
        import tensorflow as tf

        tf.logging.set_verbosity(tf.logging.ERROR)
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        """ build TensorFlow fully-connected DNN graph

        1. load/define variables for DNN
        2. build DNN
        """
        # region ---------------------------------------------------------------------- 1. load/define variables for DNN
        if self.FLAGS.features:
          self.feature_selector = self.FLAGS.features

        # start an interactive session
        config = tf.ConfigProto()

        config.gpu_options.per_process_gpu_memory_fraction=0.1
        config.gpu_options.allow_growth = True
        config.log_device_placement = False

        self.sess = tf.InteractiveSession(config=config)

        # placeholder for input variables
        self.x = tf.placeholder(tf.float32, shape=[None, self.input_size], name='X')
        self.y_ = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='y')
        self.keep_prob_input = tf.placeholder(tf.float32, name='kpi')
        self.keep_prob_hidden = tf.placeholder(tf.float32, name='kph')

        self.class_weights = tf.placeholder(tf.float32, shape=[self.num_classes], name='cw')  # only if class weighting is used
        # endregion ----------------------------------------------------------------------------------------------------

        # region ----------------------------------------------------------------------------------- dnn build functions

        def weight_variable(shape, stddev):
            """
                create weight variables for a whole layer
                using assign tensors that can be passed to
                an interactive session for weight resets
            """
            initial = tf.truncated_normal(shape, stddev=stddev)
            w = tf.Variable(initial)
            w_init = tf.assign(w, initial)
            self.initial_layer_weights.append(w_init)
            return w


        def bias_variable(shape):
            """ create bias variables for a whole layer """
            initial = tf.zeros(shape)
            return tf.Variable(initial)

        def fc_layer(x, channels_in, channels_out, stddev):
            """ create a whole layer W * x + b with ReLU transfer function """
            W = weight_variable([channels_in, channels_out], stddev)
            b = bias_variable([channels_out])
            act = tf.nn.relu(tf.matmul(x, W) + b)
            return act

        def logits_fn(x, channels_in, channels_out, stddev):
            """ create a whole output layer W * x + b """
            W = weight_variable([channels_in, channels_out], stddev)
            b = bias_variable([channels_out])
            act = tf.matmul(x, W) + b
            return act
        # endregion ----------------------------------------------------------------------------------------------------

        # region ------------------------------------------------------------------------------------------ 2. build DNN
        x_drop_inn = tf.nn.dropout(self.x, self.keep_prob_input)

        # input layer
        h_fc_prev = fc_layer(x_drop_inn, self.input_size, self.FLAGS.layers[0], stddev=1.0 / math.sqrt(float(self.input_size)))
        h_fc_prev = tf.nn.dropout(h_fc_prev, self.keep_prob_hidden)

        for l, l_size in enumerate(self.FLAGS.layers[1:]):  # create hidden layers based on command line parameters
          h_fc_prev = fc_layer(h_fc_prev, self.FLAGS.layers[l], l_size, 1.0 / math.sqrt(float(self.FLAGS.layers[l])))
          h_fc_prev = tf.nn.dropout(h_fc_prev, self.keep_prob_hidden)

        # create output layer (a softmax linear classification layer_sizes for the outputs)
        self.logits = logits_fn(h_fc_prev, self.FLAGS.layers[-1], self.num_classes, stddev=1.0 / math.sqrt(float(self.FLAGS.layers[-1])))

        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y_, logits=self.logits)
        if self.FLAGS.cw_method == 0:  # apply class weighting based on a weighting factor of a block
          weight_factor = tf.gather(self.class_weights, tf.cast(tf.argmax(self.y_, 1), tf.int32))
          avg_loss = tf.reduce_mean(cross_entropy_loss * weight_factor)
        else:
          avg_loss = tf.reduce_mean(cross_entropy_loss)  # no class weighting

        if self.FLAGS.optimizer == 'Adam':
            self.train_step = tf.train.AdamOptimizer(self.FLAGS.learning_rate).minimize(avg_loss)
        if self.FLAGS.optimizer == 'SGD':
            self.train_step = tf.train.GradientDescentOptimizer(self.FLAGS.learning_rate).minimize(avg_loss)

        self.get_prediction = tf.argmax(self.logits, 1)
        self.get_pred_prob = tf.nn.softmax(self.logits)

        self.correct_prediction_tr = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.y_, 1))
        self.accuracy_tr = tf.reduce_mean(tf.cast(self.correct_prediction_tr, tf.float32))

        # endregion

        tf.global_variables_initializer().run()  # initialize all global variables

