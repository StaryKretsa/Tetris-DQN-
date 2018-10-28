import tensorflow as tf
import numpy as np
import random




def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

"""def norm(a):
        b = np.mean(a, axis=2)
        c = np.zeros((10, 5))
        for i in range(10):
            for j in range(5):
                c[i][j]=(b[j*20+5][i*20+5] > 0)
        return c
"""
        

    
EPS = 10e-9
class DQN:
  
    def __init__(
        self,
        state_shape,
        n_actions,
        learning_rate = 1e-4,
        reward_decay = 0.9,
        epsilon = 1,
        memory_size = 1024,
        batch_size = 64,
    ):
        self.n_actions = n_actions
        self.n_states = state_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = epsilon

        self.memory_size = memory_size
        self.batch_size = batch_size
        self.review_size = 32
        self.memory = [] 
        self.cost_board = []
        self.cost_tmp = []
        self.total_loss = 0

        self.build_cnn_net()

        self.sess = tf.InteractiveSession()
#        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        self.p_loss = 1

        print("Initialize Done!")

    

    def build_cnn_net(self):
        
        self.input_layer = tf.placeholder(tf.float32, [None, 10, 5, 4])
        self.action_input = tf.placeholder(tf.float32, [None, self.n_actions])
        self.y = tf.placeholder(tf.float32, [None])
        self.keep_prob = tf.placeholder(tf.float32)
        W_conv1 = weight_variable([3, 3, 4, 32])
        b_conv1 = bias_variable([32])
        W_conv2 = weight_variable([3, 3, 32, 32])
        b_conv2 = bias_variable([32])
        W_fc1 = weight_variable([10 * 5 * 32, 1024])
        b_fc1 = bias_variable([1024])
        W_fc3 = weight_variable([1024, self.n_actions])
        b_fc3 = bias_variable([self.n_actions])

        r_input_layer = tf.reshape(self.input_layer, [-1, 10, 5, 4]) 
        h_conv1 = tf.nn.relu(conv2d(r_input_layer, W_conv1) + b_conv1)
        h_conv1_d = tf.nn.dropout(h_conv1,self.keep_prob)
        h_conv2 = tf.nn.relu(conv2d(h_conv1_d, W_conv2) + b_conv2)
        h_conv2_d = tf.nn.dropout(h_conv2,self.keep_prob)
        tf_pool2_flat = tf.contrib.layers.flatten(h_conv2_d)
        self.h_fc1 = tf.nn.relu(tf.matmul(tf_pool2_flat, W_fc1) + b_fc1)
        self.Q_value_conv = tf.matmul(self.h_fc1, W_fc3) + b_fc3

        Q_action = tf.reduce_sum(tf.multiply(self.Q_value_conv, self.action_input), 1)
        self.loss = tf.reduce_mean(tf.squared_difference(self.y, Q_action))

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    
    def memorize(self, state, action, reward, state_, done):
       
        action_select = np.zeros(self.n_actions)
        action_select[action] = 1
#        state = norm(state) 
#        state_ = norm(state_) 
        if reward > 0:
            reward = 1
        elif reward < 0:
             reward = -1
        else:
            reward = 0

        self.memory.append(
            (state,action_select,reward,state_,done)
        )

        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

        if len(self.memory) > self.batch_size:
            self.replay()

    
    
    def replay(self):
        minibatch = random.sample(self.memory, self.batch_size)

        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        done_batch = [data[4] for data in minibatch]


        Q_value_batch =  self.Q_value_conv.eval(
            feed_dict = {
                self.input_layer :  next_state_batch,
                self.keep_prob : 1
            }
        )

        y_batch = []

        for i in range(self.batch_size):
            if done_batch[i]:
                y_batch.append(reward_batch[i]) 
            else:
                y_batch.append(reward_batch[i] + self.gamma * np.max(Q_value_batch[i]))

        cost = self.loss.eval(
            feed_dict = {
                self.input_layer : state_batch,
                self.action_input : action_batch,
                self.y : y_batch,
                self.keep_prob : 1
            }
        )

        self.total_loss += cost

        if self.p_loss == 0:
            self.cost_board.append(self.total_loss / 1000)
            self.total_loss = 0
        self.p_loss = (self.p_loss + 1) % 1000

        self.cost_tmp.append(cost)

        self.optimizer.run(
            feed_dict = {
                self.input_layer : state_batch,
                self.action_input: action_batch,
                self.y : y_batch,
                self.keep_prob : 0.75
            }
        )
    
    def greedy_action(self, state):
        return np.argmax(
            self.Q_value_conv.eval(
                feed_dict = {
                    self.input_layer : [state],
                    self.keep_prob: 1
                }
            )[0]
        )

    def e_greedy(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            return self.greedy_action(state)

    
    def save_model(self, ModelName = "model"):
        save_path = self.saver.save(self.sess, ".\model\%s.ckpt"%(ModelName))
        print("Model saved in path: %s" % save_path)
    
    def load_model(self, ModelName):
        self.saver.restore(self.sess, ".\model\%s.ckpt"%(ModelName))
        print("Model restored!")

    def set_epsilon(self, new_epsilon):
        self.epsilon = new_epsilon

    def display_Q(self, state):
#        state = norm(state) 
        print("Q:",  self.Q_value_conv.eval(feed_dict = {self.input_layer: [state], self.keep_prob : 1})[0])
#        print("Q:",  self.sess.run(self.Q_value_conv,feed_dict = {self.input_layer: [state], self.keep_prob : 1})[0])


    
