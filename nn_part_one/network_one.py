import tensorflow as tf

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
        
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))            

            conv_layer2 = tf.layers.conv2d(conv_layer1, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer2: " + str(conv_layer2.get_shape()))

            conv_layer3 = tf.layers.conv2d(conv_layer2, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer3: " + str(conv_layer3.get_shape()))            

            output_layer = tf.layers.conv2d(conv_layer3, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            
            self.predictions = output_layer

            self.loss = tf.losses.absolute_difference(self.gold_output_frames, self.predictions)
            #self.loss = tf.losses.mean_squared_error(self.gold_output_frames, self.predictions)

            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)
       
            #self.accuracy = tf.metrics.accuracy(self.gold_output_frames, self.predictions)            

            # Summaries
            self.summaries = {"training": tf.summary.merge([tf.summary.scalar("train/loss", self.loss)])}
            
            for dataset in ["validation", "test"]:
                self.summaries[dataset] = tf.summary.scalar(dataset+"/loss", self.loss)

            #init = tf.global_variables_initializer()
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.session.run(init)

            self.session.graph.finalize()           
            self.train_writer.add_graph(self.session.graph)
            print("================")

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def train(self, input_frames, output_frames, run_summaries=False, run_metadata=False):
        targets = [self.train_step, self.loss, self.predictions]
        args = {"feed_dict": {self.input_frames: input_frames, self.gold_output_frames: output_frames}}
        
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
         args = {"feed_dict": {self.input_frames: input_frames, self.gold_output_frames: output_frames}}
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
