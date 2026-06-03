from agents.base_agent import BaseAgent
import numpy as np
import tensorflow as tf

class PPOAgent(BaseAgent):
    def __init__(self, obs_dim: int, action_dim: int , config: dict):
        super().__init__(obs_dim, action_dim, config)

        # Extracting hyperparameters from the dictionary
        self.lr = self.config.get('lr', 1.0e-3)
        self.gamma = self.config.get('gamma', 0.99)
        self.lam = self.config.get('lam', 0.95)
        self.eps_clip = self.config.get('eps_clip', 0.2)
        self.n_epochs = self.config.get('n_epochs', 4)
        self.n_steps = self.config.get('n_steps', 128)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.max_updates = self.config.get('max_updates', 10000)
        self.update_count = 0

        # Building actor and critic network
        # Same networks of a2c_agent.py
        self.actor = self._build_actor()
        self.critic = self._build_critic()

        # For the PPO Network we need two more buffer compared to the A2C
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_states = []
        self.done_buffer = []

        # Again, one optimizer is sufficient, since the final loss is one and only (total loss)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-5)


        # Used to calculate the ratio, for every rollout, each action it's sampled using the logs probabilities
        # r_t = exp(log_prob_new - log_prob_old)
        # Needed to understand how much the policy has changed
        self.log_probs_old_buffer = []

        # V(s) values calculated by the critic network during the Rollout
        # GAE use them (V(s) and V(s + 1) for each step)
        # delta_t = r_t + gamma V(s + 1) - V(s)
        self.values_buffer = []

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

    # return also other to float value to store in their respective buffer
    def act(self, observation: np.ndarray , deterministic = False) -> tuple[int, float, float]:
        obs_tensor = tf.convert_to_tensor(observation, dtype=tf.float32)
        obs_tensor = tf.expand_dims(obs_tensor, axis=0)

        action_probabilities = self.actor(obs_tensor)
        # Same state, the value of the critic is returned
        value = float(self.critic(obs_tensor)[0, 0])

        if not deterministic:
            action_selected = tf.random.categorical(tf.math.log(action_probabilities +1e-8), num_samples=1)
        else:
            action_selected = tf.argmax(action_probabilities, axis=1)

        action = int(action_selected.numpy().flat[0])

        # e.g. action = 3 and action_dim = 5
        # [0, 0, 0, 1, 0]
        # Only the log prob of the action
        mask = tf.one_hot([action], self.action_dim)
        log_prob = float(tf.reduce_sum(mask * tf.math.log(action_probabilities +1e-8)))

        return action, log_prob, value

    def update(self, *args, **kwargs):
        if len(self.obs_buffer) < self.n_steps:
            return

        states = tf.convert_to_tensor(self.obs_buffer, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer, dtype=tf.int32)
        rewards = np.array(self.reward_buffer, dtype=np.float32)
        dones = np.array(self.done_buffer, dtype=np.float32)
        old_log_probs = tf.convert_to_tensor(self.log_probs_old_buffer, dtype=tf.float32)
        old_values = np.array(self.values_buffer, dtype=np.float32)

        # Bootstrap last value if episode not done
        last_next = tf.convert_to_tensor(self.next_states[-1], dtype=tf.float32)
        last_next = tf.expand_dims(last_next, axis=0)
        last_value = float(self.critic(last_next)[0, 0]) if not self.done_buffer[-1] else 0.0

        # GAE computation
        advantages = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = old_values[t + 1] if t + 1 < len(old_values) else last_value
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - old_values[t]
            gae = delta + self.gamma * self.lam * (1.0 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        advantages_tensor = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns_tensor = tf.convert_to_tensor(returns, dtype=tf.float32)
        old_values_tensor = tf.convert_to_tensor(old_values, dtype=tf.float32)

        for _ in range(self.n_epochs):
            with tf.GradientTape() as tape:
                probs = self.actor(states)
                values = tf.squeeze(self.critic(states), axis=1)

                mask = tf.one_hot(actions, self.action_dim)
                new_log_probs = tf.reduce_sum(mask * tf.math.log(probs + 1e-8), axis=1)

                # PPO clipped surrogate loss
                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages_tensor, clipped_ratio * advantages_tensor))

                # Value clipping
                values_clipped = old_values_tensor + tf.clip_by_value(
                    values - old_values_tensor, -self.eps_clip, self.eps_clip
                )
                critic_loss = tf.reduce_mean(tf.maximum(
                    tf.square(returns_tensor - values),
                    tf.square(returns_tensor - values_clipped)
                ))

                entropy = -tf.reduce_mean(tf.reduce_sum(probs * tf.math.log(probs + 1e-8), axis=1))

                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

            trainable_vars = self.actor.trainable_variables + self.critic.trainable_variables
            gradients = tape.gradient(total_loss, trainable_vars)
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.next_states.clear()
        self.done_buffer.clear()
        self.log_probs_old_buffer.clear()
        self.values_buffer.clear()

        # LR linear decay
        self.update_count += 1
        frac = max(0.0, 1.0 - self.update_count / self.max_updates)
        self.optimizer.learning_rate.assign(max(self.lr * frac, 1e-6))


    def store_experience(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, log_prob_old: float, value: float ):
        self.obs_buffer.append(state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_states.append(next_state)
        self.done_buffer.append(done)
        self.log_probs_old_buffer.append(log_prob_old)
        self.values_buffer.append(value)

    def save(self, path: str) -> None:
      self.actor.save_weights(path + "_actor_ppo.weights.h5")
      self.critic.save_weights(path + "_critic_ppo.weights.h5")

    def load(self, path: str) -> None:
      self.actor.load_weights(path + "_actor_ppo.weights.h5")
      self.critic.load_weights(path + "_critic_ppo.weights.h5")
