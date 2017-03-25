import tensorflow as tf

# Classes #
class Network:
    def __init__(self, frame_height, frame_width, exp_name):
        self.frame_height = frame_height
        self.frame_width = frame_width
        graph = tf.Graph()
        self.session = tf.Session(graph = graph)        
    
        self.train_writer = tf.summary.FileWriter("./logs/network-one/train/%s" % exp_name, self.session.graph)
        self.test_writer = tf.summary.FileWriter("./logs/network-one/test/%s" % exp_name)

    def construct(self):
        print("================")
        with self.session.graph.as_default():
            self.input_frames = tf.placeholder(tf.float32, [None, self.frame_height, self.frame_width, 2])
            print("\tself.input_frames: " + str(self.input_frames.get_shape()))            
            self.gold_output_frames = tf.placeholder(tf.int32, [None, self.frame_height, self.frame_width, 1])
            print("\tself.output_frames: " + str(self.gold_output_frames.get_shape()))   
        
            # crossentropy_loss model #
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [5, 5], 1, padding="SAME")
            self.__print_tensor_shape(conv_layer1, "\tconv_layer1")                                                                                                       

            output_layer = tf.layers.conv2d(conv_layer1, 2, [1, 1], 1, padding="SAME")
            self.__print_tensor_shape(output_layer, "\toutput_layer")

            reshaped_output = tf.reshape(output_layer, [-1, 2])
            reshaped_gold_output_frames = tf.reshape(self.gold_output_frames, [-1])
            self.loss = tf.losses.sparse_softmax_cross_entropy(reshaped_gold_output_frames, reshaped_output)

            softmax_layer = tf.nn.softmax(output_layer) # (?, 20, 20, 2)
            target_pixel_predictions = tf.argmax(softmax_layer, -1) # (?, 20, 20)
            self.predictions = target_pixel_predictions
            
            target_pixel_predictions = tf.reshape(target_pixel_predictions, [-1])
            self.accuracy = tf.metrics.accuracy(reshaped_gold_output_frames, target_pixel_predictions)
            
            ##############################################
                     
            self.global_step = tf.Variable(0, dtype=tf.int64, trainable=False)
            self.train_step = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            # Summaries            
            self.summaries = {"training": tf.summary.merge([tf.summary.scalar("train/loss", self.loss),
                                                            tf.summary.scalar("train/accuracy", self.accuracy[1])])}
            
            for dataset in ["validation", "test"]:
                self.summaries[dataset] = tf.summary.merge([tf.summary.scalar(dataset+"/loss", self.loss),
                                                            tf.summary.scalar(dataset+"/accuracy", self.accuracy[1])])

            #init = tf.global_variables_initializer()
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            self.session.run(init)

            self.session.graph.finalize()           
            self.train_writer.add_graph(self.session.graph)
            print("================")    

    @property
    def training_step(self):
        return self.session.run(self.global_step)

    def __print_tensor_shape(self, tensor, tensor_name):
        print("%s: %s" % (tensor_name, str(tensor.get_shape())))

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
         targets = [self.predictions, self.loss, self.accuracy]

         if run_summaries:
            targets.append(self.summaries[dataset])

         results = self.session.run(targets, **args)

         if run_summaries:
            summary = results[-1]            
            self.test_writer.add_summary(summary, self.training_step)
         
         predictions = results[0]
         loss = results[1]
         accuracy = results[2]
         return loss, predictions, accuracy
