from agents.base_agent import BaseAgent
import numpy as np
import tensorflow as tf


class MAPPOAgent(BaseAgent):
    def __init__(self, obs_dim: int, action_dim: int, global_obs_dim: int, config: dict):
        super().__init__(obs_dim, action_dim, config)

        self.global_obs_dim = global_obs_dim
        self.lr = self.config.get('lr', 3e-4)
        self.gamma = self.config.get('gamma', 0.99)
        self.lam = self.config.get('lam', 0.95)
        self.eps_clip = self.config.get('eps_clip', 0.2)
        self.n_epochs = self.config.get('n_epochs', 4)
        self.n_steps = self.config.get('n_steps', 256)
        self.entropy_coef = self.config.get('entropy_coef', 0.01)
        self.value_coef = self.config.get('value_coef', 0.5)
        self.max_updates = self.config.get('max_updates', 10000)
        self.update_count = 0

        # Actor: decentralized, takes only local observation
        self.actor = self._build_actor()
        # Critic: centralized, takes global state (all agents' observations concatenated)
        self.critic = self._build_critic()

        self.obs_buffer = []
        self.global_states_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.next_global_states_buffer = []
        self.done_buffer = []
        self.log_probs_old_buffer = []
        self.values_buffer = []

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, epsilon=1e-5)

    def _build_actor(self):
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.obs_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='softmax')
        ])

    def _build_critic(self):
        # Larger capacity than actor: centralized critic processes the full global state
        return tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.global_obs_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])

    def act(self, observation: np.ndarray, deterministic=False, global_state=None) -> tuple[int, float, float]:
        obs_tensor = tf.expand_dims(tf.convert_to_tensor(observation, dtype=tf.float32), axis=0)
        action_probabilities = self.actor(obs_tensor)

        # During evaluation global_state is not provided: value is unused so we return 0.0
        if global_state is not None:
            gs_tensor = tf.expand_dims(tf.convert_to_tensor(global_state, dtype=tf.float32), axis=0)
            value = float(self.critic(gs_tensor)[0, 0])
        else:
            value = 0.0

        if not deterministic:
            action_selected = tf.random.categorical(tf.math.log(action_probabilities + 1e-8), num_samples=1)
        else:
            action_selected = tf.argmax(action_probabilities, axis=1)

        action = int(action_selected.numpy().flat[0])
        mask = tf.one_hot([action], self.action_dim)
        log_prob = float(tf.reduce_sum(mask * tf.math.log(action_probabilities + 1e-8)))

        return action, log_prob, value

    def store_experience(self, state: np.ndarray, action: int, reward: float,
                         next_state: np.ndarray, done: bool, log_prob_old: float,
                         value: float, global_state: np.ndarray, next_global_state: np.ndarray):
        self.obs_buffer.append(state)
        self.global_states_buffer.append(global_state)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.next_global_states_buffer.append(next_global_state)
        self.done_buffer.append(done)
        self.log_probs_old_buffer.append(log_prob_old)
        self.values_buffer.append(value)

    def clear_buffers(self):
        self.obs_buffer.clear()
        self.global_states_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.next_global_states_buffer.clear()
        self.done_buffer.clear()
        self.log_probs_old_buffer.clear()
        self.values_buffer.clear()

    def update(self, *args, force=False, **kwargs):
        min_batch = 32
        if len(self.obs_buffer) == 0:
            return
        if not force and len(self.obs_buffer) < self.n_steps:
            return
        if force and len(self.obs_buffer) < min_batch:
            self.clear_buffers()
            return

        states = tf.convert_to_tensor(self.obs_buffer, dtype=tf.float32)
        global_states = tf.convert_to_tensor(self.global_states_buffer, dtype=tf.float32)
        actions = tf.convert_to_tensor(self.action_buffer, dtype=tf.int32)
        rewards = np.array(self.reward_buffer, dtype=np.float32)
        dones = np.array(self.done_buffer, dtype=np.float32)
        old_log_probs = tf.convert_to_tensor(self.log_probs_old_buffer, dtype=tf.float32)
        old_values = np.array(self.values_buffer, dtype=np.float32)

        last_next_gs = tf.expand_dims(
            tf.convert_to_tensor(self.next_global_states_buffer[-1], dtype=tf.float32), axis=0
        )
        last_value = float(self.critic(last_next_gs)[0, 0]) if not self.done_buffer[-1] else 0.0

        # GAE with centralized value estimates
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
                # Actor uses local observations (decentralized execution)
                probs = self.actor(states)
                # Critic uses global state (centralized training)
                values = tf.squeeze(self.critic(global_states), axis=1)

                mask = tf.one_hot(actions, self.action_dim)
                new_log_probs = tf.reduce_sum(mask * tf.math.log(probs + 1e-8), axis=1)

                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped_ratio = tf.clip_by_value(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                actor_loss = -tf.reduce_mean(
                    tf.minimum(ratio * advantages_tensor, clipped_ratio * advantages_tensor)
                )

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
        self.global_states_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.next_global_states_buffer.clear()
        self.done_buffer.clear()
        self.log_probs_old_buffer.clear()
        self.values_buffer.clear()

        self.update_count += 1
        frac = max(0.0, 1.0 - self.update_count / self.max_updates)
        self.optimizer.learning_rate.assign(max(self.lr * frac, 1e-6))

    def save(self, path: str) -> None:
        self.actor.save_weights(path + "_actor_mappo.weights.h5")
        self.critic.save_weights(path + "_critic_mappo.weights.h5")

    def load(self, path: str) -> None:
        self.actor.load_weights(path + "_actor_mappo.weights.h5")
        self.critic.load_weights(path + "_critic_mappo.weights.h5")
