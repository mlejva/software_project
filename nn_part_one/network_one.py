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
            self.gold_output_frames = tf.placeholder(tf.float32, [None, self.frame_height, self.frame_width, 1])
            print("\tself.output_frames: " + str(self.gold_output_frames.get_shape()))
        
            # base model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))            

            conv_layer2 = tf.layers.conv2d(conv_layer1, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer2: " + str(conv_layer2.get_shape()))

            conv_layer3 = tf.layers.conv2d(conv_layer2, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer3: " + str(conv_layer3.get_shape()))                                                                   

            output_layer = tf.layers.conv2d(conv_layer3, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-7 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))            

            conv_layer2 = tf.layers.conv2d(conv_layer1, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer2: " + str(conv_layer2.get_shape()))

            conv_layer3 = tf.layers.conv2d(conv_layer2, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer3: " + str(conv_layer3.get_shape()))

            conv_layer4 = tf.layers.conv2d(conv_layer3, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer4: " + str(conv_layer4.get_shape()))                   

            conv_layer5 = tf.layers.conv2d(conv_layer4, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer5: " + str(conv_layer5.get_shape()))                   

            conv_layer6 = tf.layers.conv2d(conv_layer5, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer6: " + str(conv_layer6.get_shape()))                   

            output_layer = tf.layers.conv2d(conv_layer6, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # base-channels-48 model #   
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))            

            conv_layer2 = tf.layers.conv2d(conv_layer1, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer2: " + str(conv_layer2.get_shape()))

            conv_layer3 = tf.layers.conv2d(conv_layer2, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer3: " + str(conv_layer3.get_shape()))                                                                   

            output_layer = tf.layers.conv2d(conv_layer3, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-7-channels-48 model #            
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))            

            conv_layer2 = tf.layers.conv2d(conv_layer1, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer2: " + str(conv_layer2.get_shape()))

            conv_layer3 = tf.layers.conv2d(conv_layer2, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer3: " + str(conv_layer3.get_shape()))

            conv_layer4 = tf.layers.conv2d(conv_layer3, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer4: " + str(conv_layer4.get_shape()))                   

            conv_layer5 = tf.layers.conv2d(conv_layer4, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer5: " + str(conv_layer5.get_shape()))                   

            conv_layer6 = tf.layers.conv2d(conv_layer5, 48, [3, 3], 1, padding="SAME")
            print("\tconv_layer6: " + str(conv_layer6.get_shape()))                   

            output_layer = tf.layers.conv2d(conv_layer6, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # base-channels-9 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 9, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))            

            conv_layer2 = tf.layers.conv2d(conv_layer1, 9, [3, 3], 1, padding="SAME")
            print("\tconv_layer2: " + str(conv_layer2.get_shape()))

            conv_layer3 = tf.layers.conv2d(conv_layer2, 9, [3, 3], 1, padding="SAME")
            print("\tconv_layer3: " + str(conv_layer3.get_shape()))                                                                   

            output_layer = tf.layers.conv2d(conv_layer3, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''                        

            # conv-2-channels-8 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 8, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer1, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-2-channels-4 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 4, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer1, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-2-channels-18 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 18, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer1, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-2-channels-24 model #
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer1, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))

            # conv-2-channels-24-kernel-1-1 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [1, 1], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer1, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-2-channels-24-kernel-2-2 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [2, 2], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer1, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-3-channels-24-kernel-1-3_3-1 model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [1, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                         

            conv_layer2 = tf.layers.conv2d(conv_layer1, 24, [3, 1], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer2, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''

            # conv-2-channels-24-loss-abs-diff model #
            '''
            conv_layer1 = tf.layers.conv2d(self.input_frames, 24, [3, 3], 1, padding="SAME")
            print("\tconv_layer1: " + str(conv_layer1.get_shape()))                                                                                           

            output_layer = tf.layers.conv2d(conv_layer1, 1, [1, 1], 1, padding="SAME")
            print("\toutput_layer: " + str(output_layer.get_shape()))
            '''                    
        
            self.predictions = output_layer

            #target_pixel = tf.argmax(output_layer, -1)
            #self.predictions = tf.constant

                        
            self.loss = tf.losses.mean_squared_error(self.gold_output_frames, self.predictions)
            #self.loss = tf.losses.absolute_difference(self.gold_output_frames, self.predictions)

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
