import torch
import torch.nn.functional as F
import numpy as np
import os
from cnn_health_hunger import CNNHealthHunger, detect_status
from config import DEVICE, BATCH_SIZE, GAMMA, STATE_DICT_PATH, CNN_STATE_DICT_PATH
from PIL import Image

class Optimizer:

  FILENAME = 'policyweights.dat'

  def __init__(self, policy_net, target_net, optimizer, action_spaces):
      self.policy_net = policy_net
      self.target_net = target_net
      self.optimizer = optimizer
      self.action_spaces = action_spaces
      self.policy_filename = None
      self.initialize_status_sensors()


  def initialize_status_sensors(self):
      self.healthy_detector = CNNHealthHunger()
      self.pain_detector = CNNHealthHunger()
      self.full_detector = CNNHealthHunger()
      self.starving_detector = CNNHealthHunger()
      healthy_weights = os.path.join(CNN_STATE_DICT_PATH, 'health_healthy_unhealthy_params.dat')
      pain_weights = os.path.join(CNN_STATE_DICT_PATH, 'health_pain_no_pain_params.dat')
      full_weights = os.path.join(CNN_STATE_DICT_PATH, 'hunger_full_not_full_params.dat')
      starving_weights = os.path.join(CNN_STATE_DICT_PATH, 'hunger_starving_fed_params.dat')
      self.healthy_detector.load_state_dict(torch.load(healthy_weights))
      self.pain_detector.load_state_dict(torch.load(pain_weights))
      self.full_detector.load_state_dict(torch.load(full_weights))
      self.starving_detector.load_state_dict(torch.load(starving_weights))
      
  def calculate_reward(self, image_data, reward, inventory):
      new_reward = reward
      average_brightness = np.mean(image_data)
      if average_brightness > 100:
        new_reward = new_reward + 0.5
      if average_brightness < 70:
        new_reward = new_reward - 1
      
      healthy = False
      pain = False
      full = False
      starving = False

      im = Image.fromarray(np.array(image_data).astype(np.uint8))
      health_im = self.healthy_detector.crop_image(im, 'health')
      food_im = self.full_detector.crop_image(im, 'food')
      if detect_status(health_im, self.healthy_detector):
        healthy = True
        new_reward = new_reward + 1
      
      if detect_status(health_im, self.pain_detector):
        pain = True
        new_reward = new_reward - 10
      
      if detect_status(food_im, self.full_detector):
        full = True
        new_reward = new_reward + 1

      if detect_status(food_im, self.starving_detector):
        starving = True
        new_reward = new_reward - 5

      return [torch.tensor([[new_reward for i in range(len(self.action_spaces))]], device=DEVICE), healthy, pain, full, starving, average_brightness]


  def optimize_model(self, memory):
      if len(memory) < BATCH_SIZE:
          return
    
      transitions = memory.sample(BATCH_SIZE)
      # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
      # detailed explanation). This converts batch-array of Transitions
      # to Transition of batch-arrays.
      batch = memory.Transition(*zip(*transitions))

      # Compute a mask of non-final states and concatenate the batch elements
      # (a final state would've been the one after which simulation ended)
      non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=DEVICE, dtype=torch.bool)
      non_final_next_states = torch.cat([s for s in batch.next_state
                                                  if s is not None])
    

      state_batch = torch.cat(batch.state)
      action_batch = torch.cat(batch.action)
      reward_batch = torch.cat(batch.reward)

      # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
      # columns of actions taken. These are the actions which would've been taken
      # for each batch state according to policy_net
      state_action_values = self.policy_net(state_batch).gather(1, action_batch)

      # Compute V(s_{t+1}) for all next states.
      # Expected values of actions for non_final_next_states are computed based
      # on the "older" target_net; selecting their best reward with max(1)[0].
      # This is merged based on the mask, such that we'll have either the expected
      # state value or 0 in case the state was final.
      next_state_values = torch.zeros([BATCH_SIZE, len(self.action_spaces)], device=DEVICE)
      action_tensor = self.target_net(non_final_next_states)
      next_state_values[non_final_mask] = self.process_state_actions(action_tensor.detach()).type(torch.FloatTensor)
      # Compute the expected Q values
      expected_state_action_values = (next_state_values * GAMMA) + reward_batch
      # Compute Huber loss
      loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
      # Optimize the model
      self.optimizer.zero_grad()
      loss.backward()
      for param in self.policy_net.parameters():
          param.grad.data.clamp_(-1, 1)
      self.optimizer.step()


  def process_state_actions(self, action_tensor):
      placeholder = action_tensor.numpy()[0]
      action_probabilities = []
      count = 0
      for group, action_items in enumerate(self.action_spaces):
        action_probabilities.append([])
        for index in range(count, count + action_items):
          action_probabilities[group].append(placeholder[index])
          count = count + 1

      choices = []
      for discrete_action_space in action_probabilities:
        choices.append(np.argmax(discrete_action_space))
      choice_tensor = torch.tensor([choices])  
      return choice_tensor


  def update_target(self):
      self.target_net.load_state_dict(self.policy_net.state_dict())

  def save_policy_weights(self):
      filename = os.path.join(STATE_DICT_PATH, self.FILENAME)
      self.policy_net.version = self.policy_net.version + 1
      torch.save(self.policy_net.state_dict(), filename)
      torch.save(self.policy_net.state_dict(), os.path.join(STATE_DICT_PATH,"version_" + str(self.policy_net.version) + "_" + self.FILENAME))