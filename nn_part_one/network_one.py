import tensorflow as tf
import tensorflow.contrib.layers as tf_layers

# Classes #
class Network:
    def __init__(self, frame_height, frame_width):
        self.frame_height = frame_height
        self.frame_width = frame_width
        graph = tf.Graph()
        self.session = tf.Session(graph = graph)        
    
        self.train_writer = tf.summary.FileWriter("./logs/network-one/train", self.session.graph)
        self.test_writer = tf.summary.FileWriter("./logs/network-one/test")

    def construct(self):
        print("================")
        with self.session.graph.as_default():
            self.input_frames = tf.placeholder(tf.float32, [None, self.frame_height, self.frame_width, 2])
            print("\tself.input_frames: " + str(self.input_frames.get_shape()))            
            self.gold_output_frames = tf.placeholder(tf.float32, [None, self.frame_height, self.frame_width, 1])
            print("\tself.output_frames: " + str(self.gold_output_frames.get_shape()))
        
            conv_layer1 = tf_layers.convolution2d(self.input_frames, 24, [3, 3], 1, padding="SAME", normalizer_fn=tf_layers.batch_norm)
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))            

            conv_layer2 = tf_layers.convolution2d(conv_layer1, 24, [3, 3], 1, padding="SAME", normalizer_fn=tf_layers.batch_norm)
            print("\tconv_layer2: " + str(conv_layer2.get_shape()))

            conv_layer3 = tf_layers.convolution2d(conv_layer2, 24, [3, 3], 1, padding="SAME", normalizer_fn=tf_layers.batch_norm)
            print("\tconv_layer3: " + str(conv_layer3.get_shape()))

            conv_layer4 = tf_layers.convolution2d(conv_layer3, 1, [1, 1], 1, padding="SAME", normalizer_fn=tf_layers.batch_norm)
            print("\tconv_layer4: " + str(conv_layer4.get_shape()))

            output_layer = conv_layer4
            print("\toutput_layer: " + str(output_layer.get_shape()))

            self.predictions = output_layer
            self.loss = tf.losses.absolute_difference(self.gold_output_frames, self.predictions)
            tf.summary.scalar("loss function", self.loss)

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            self.accuracy = tf.metrics.accuracy(self.predictions, self.gold_output_frames)
            #tf.summary.scalar("accuracy", self.accuracy)

            # Summaries
            self.summaries = tf.summary.merge_all()            

            init = tf.global_variables_initializer()
            self.session.run(init)

            self.session.graph.finalize()           
            self.train_writer.add_graph(self.session.graph)
            print("================")

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, input_frames, output_frames, run_summaries=False, run_metadata=False):
        targets = [self.train_step, self.loss]
        args = {"feed_dict": {self.input_frames: input_frames, self.gold_output_frames: output_frames}}
        
        if run_summaries:
            targets.append(self.summaries)
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
        return loss
        #print(results[1])

    def evaluate(self, input_frames, output_frames):
         args = {"feed_dict": {self.input_frames: input_frames, self.output_frames: output_frames}}
         targets = [self.accuracy]
         results = self.session.run(targets, **args)
         return results[0]

    #def run(self, input_frames, output_frames):
    #    print(self.session.run(self.input_frames, {self.input_frames: input_frames, self.output_frames: output_frames}))


        #print(self.session.run(input_frames, feed_dict={self.input_frame_fst: input_frames[0], self.input_frame_fst: input_frames[1], self.output_frame: output_frame}))
