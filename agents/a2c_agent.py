from agents.base_agent import BaseAgent
import tensorflow as tf
import numpy as np

class A2CAgent(BaseAgent):
    def __init__(self, obs_dim: int, action_dim: int , config: dict):
        super().__init__(obs_dim, action_dim, config)
        
        # Reading A2C specific hyperparameters from the config  
        self.lr = self.config.get('lr', 1.0e-3)
        self.gamma = self.config.get('gamma', 0.99)
        self.n_steps = self.config.get('n_steps', 5)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.value_coef = self.config.get('value_coef', 0.5)
        
        # Building the actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        #The optimizer is shared between the actor and critic, since they are trained together
        #One final loss function will be used to compute the models' update
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr , epsilon = 1e-5)
    
        #Using 5 different buffers to store the experience
        #Updating the networks every n_steps
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_states = []
        self.done_buffer = []

    #Policy Network, give probability distribution of the actions given the current observation
    def _build_actor(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])
        return model
    
    #Value Network, estimate how good the current state is, that's why has a linear activation funcion (every result is valid)
    def _build_critic(self):
       model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
       return model

    #NOTE: I want the code to be agnostic, so the training and the evaluator would be the same for each agent
    #that's why this method returns a tuple[int,float,float] when just the int (action_selected) is needed
    #same goes below with store_experience method (the last two parameters setted to None)

    def act(self, observation: np.ndarray, deterministic = False ) -> tuple[int, float]:
        
        # Converting observation array to a float32 tensor + batch dimension
        obs_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        obs_tensor = tf.expand_dims(obs_tensor, axis=0)  # (obs,) -> (1, obs)

        # Letting the actor determines the probabily distribution

        #Expected dimension: (1, action_dim)
        action_probabilities = self.actor(obs_tensor)

        if not deterministic: 
            #Exploration needed, sampling an action from the distribution
            action_selected = tf.random.categorical(tf.math.log(action_probabilities + 1e-8), num_samples=1)
        
        else:
            #Inference phase, argmax (not stochastic anymore -> deterministic)
            action_selected = tf.argmax(action_probabilities, axis=1)
        

        return int(action_selected.numpy().flat[0]), None, None
        



    def update(self, *args, **kwargs):
        if not self.obs_buffer:
            return

        # Firslty, converting the buffers to tensors, used later for computing the loss function
        states = tf.convert_to_tensor(self.obs_buffer, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer, dtype=tf.int32)

        #If the episode is ended, R = 0
        if self.done_buffer[-1]:
            R = 0.0
        else:
            # Bootstrap from last next_state if episode isn't over
            last_next = tf.convert_to_tensor(self.next_states[-1], dtype=tf.float32)
            last_next = tf.expand_dims(last_next, axis=0)                             
            R = float(self.critic(last_next)[0, 0])
        
        returns = []       

        # Walk backwards through rewards to compute discounted n-step returns     
        # (1 - done) zeroes out R when an intermediate step ends the episode
        for reward, done in zip(reversed(self.reward_buffer), reversed(self.done_buffer)):
            R = reward + self.gamma * R * (1.0 - float(done))                         
            returns.append(R)                                                         
        
        returns.reverse()                           
        # Q(St​,At​) estimation                                  
        returns = tf.convert_to_tensor(returns, dtype=tf.float32) 

        with tf.GradientTape() as tape:
              probs = self.actor(states)
              # V(St​) estimation 
              values = tf.squeeze(self.critic(states), axis=1)                  

              # Advantage = A(St​,At​) = Q(St​,At​) − V(St​)
              # Calculating if the action is better or worse than expected   
              # tf.stop_gradient necessary to force the update only on the probability distribution                                                         
              advantage = tf.stop_gradient(returns - values)                    
                                                                                
              mask = tf.one_hot(actions, self.action_dim)                       
              log_probs = tf.reduce_sum(mask * tf.math.log(probs + 1e-8), axis=1)                                                                       

              #Negative --- Gradient Ascent    
              actor_loss = -tf.reduce_mean(log_probs * advantage)      
              
              # critic network will depend on a different loss value
              # Gradient Descent, it works as a Linear Regression model
              critic_loss = tf.reduce_mean(tf.square(returns - values))
              entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))
                                                                                
              total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # Clip gradients to avoid instability, then update both networks
        trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
        gradients = tape.gradient(total_loss, trainable_vars)                 
        gradients, _ = tf.clip_by_global_norm(gradients, 0.5)

        #Applying the updates on the models parameters
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))        

        # Clear buffers for next rollout                                                                        
        self.obs_buffer, self.action_buffer, self.reward_buffer, self.next_states, self.done_buffer = [], [], [], [], []

    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, log_prob_old=None, value=None):
        self.obs_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_states.append(next_state)
        self.done_buffer.append(done)
        
    
    def save(self, path: str) -> None:                                            
      self.actor.save_weights(path + "_actor_a2c.weights.h5")
      self.critic.save_weights(path + "_critic_a2c.weights.h5")                     
                                                                                
    def load(self, path: str) -> None:                                            
      self.actor.load_weights(path + "_actor_a2c.weights.h5")                       
      self.critic.load_weights(path + "_critic_a2c.weights.h5")
