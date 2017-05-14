import tensorflow as tf
import numpy as np

# Classes
class Network:
    def __init__(self, frame_height, frame_width, exp_name):
        self.frame_height = frame_height
        self.frame_width = frame_width
        graph = tf.Graph()
        self.session = tf.Session(graph=graph)

        self.train_writer = tf.summary.FileWriter("./logs/network_modif_1/train/%s" % exp_name,
                                                  self.session.graph)
        self.test_writer = tf.summary.FileWriter("./logs/network_modif_1/test/%s" % exp_name)

    def construct(self):
        with self.session.graph.as_default():
            # Network input & gold output
            self.input_frames = tf.placeholder(tf.float32,
                                               [None, self.frame_height, self.frame_width, 2])
            self.gold_frames = tf.placeholder(tf.float32,
                                              [None, self.frame_height, self.frame_width, 1])

            # Hidden layers
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [1, 3], 1, padding="SAME")
            self.__print_tensor_shape(conv_layer1, "\tconv_layer1")

            conv_layer2 = tf.layers.conv2d(conv_layer1, 24, [3, 1], 1, padding="SAME")
            self.__print_tensor_shape(conv_layer2, "\tconv_layer2")

            conv_layer3 = tf.layers.conv2d(conv_layer2, 1, [1, 1], 1, padding="SAME")
            self.__print_tensor_shape(conv_layer3, "\tconv_layer3")

            output_layer = conv_layer3

            gold = tf.sigmoid(self.gold_frames)
            self.loss = tf.losses.sigmoid_cross_entropy(gold, output_layer)
            #self.loss = tf.losses.mean_pairwise_squared_error(gold, output_layer)

            self.predictions = tf.sigmoid(output_layer)

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
            #######

            # Summaries
            self.summaries = {"training": tf.summary.merge([tf.summary.scalar("train/loss", self.loss)])}

            for dataset in ["validation", "test"]:
                self.summaries[dataset] = tf.summary.merge([tf.summary.scalar(dataset + "/loss", self.loss)])

            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.session.run(init)

            self.session.graph.finalize()
            self.train_writer.add_graph(self.session.graph)

    # Properties
    @property
    def training_step(self):
        return self.session.run(self.global_step)

     # Private methods
    def __print_tensor_shape(self, tensor, tensor_name):
        print("%s: %s" % (tensor_name, str(tensor.get_shape())))

    # Public methods
    def log_scalar_test(self, value, name, epoch, dataset):
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=dataset + "/" + name, simple_value=value)
        ])
        self.test_writer.add_summary(summary, epoch)

    def train(self, input_frames, output_frames, run_summaries=False, run_metadata=False):
        targets = [self.train_step, self.loss, self.predictions]
        args = {"feed_dict": {self.input_frames: input_frames, self.gold_frames: output_frames}}

        if run_summaries:
            targets.append(self.summaries["training"])
        if run_metadata:
            args["options"] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            args["run_metadata"] = tf.RunMetadata()

        results = self.session.run(targets, **args)

        if run_summaries:
            summary = results[-1]
            self.train_writer.add_summary(summary, self.training_step - 1)
        if run_metadata:
            self.train_writer.add_run_metadata(args["run_metadata"], "step%d" % (self.training_step - 1))

        loss = results[1]
        predictions = results[2]
        return loss, predictions

    def evaluate(self, dataset, input_frames, output_frames, run_summaries=False):
         args = {"feed_dict": {self.input_frames: input_frames, self.gold_frames: output_frames}}
         targets = [self.predictions, self.loss]

         if run_summaries:
            targets.append(self.summaries[dataset])

         results = self.session.run(targets, **args)

         if run_summaries:
            summary = results[-1]
            self.test_writer.add_summary(summary, self.training_step)

         predictions = results[0]
         loss = results[1]
         return loss, predictions
