from flask import Flask, request, send_from_directory, make_response
import torch
import torch.optim as optim
from dqn import DQN
from dqn_replay import ReplayMemory
from dqn_optimizer import Optimizer
from config import CONVOLUTION_WIDTH, CONVOLUTION_HEIGHT, TARGET_UPDATE, STATE_DICT_PATH, REPLAY_MEMORY, ACTION_SPACES, MONGO_CLIENT, MONGO_DBNAME, HOST, DEVICE
import numpy as np 
import os
import pymongo

app = Flask(__name__)
number_outputs = np.sum(ACTION_SPACES)
policy_dqn = DQN(CONVOLUTION_WIDTH, CONVOLUTION_HEIGHT, number_outputs).to(DEVICE)
target_dqn = DQN(CONVOLUTION_WIDTH, CONVOLUTION_HEIGHT, number_outputs).to(DEVICE)
memory = ReplayMemory(REPLAY_MEMORY)
optimizer_algorithm = optim.RMSprop(policy_dqn.parameters())
optimizer = Optimizer(policy_dqn, target_dqn, optimizer_algorithm, ACTION_SPACES)
mongo_client = pymongo.MongoClient("mongodb://" + MONGO_CLIENT)

if(os.path.isfile(os.path.join(STATE_DICT_PATH, optimizer.FILENAME))):
  policy_dqn.load_state_dict(torch.load(os.path.join(STATE_DICT_PATH, optimizer.FILENAME)))
  policy_dqn.eval()

target_dqn.load_state_dict(policy_dqn.state_dict())
target_dqn.eval()
print("Initializing...")

@app.route('/optimize', methods=['POST'])
def optimize():
  data = request.json
  image_data = data["image"]
  reward = data["reward"]
  step = data["step"]
  trial = data["trial"]
  episode = data["episode"]
  action = data["action"]
  last_state = data["last_state"]
  next_state = data["next_state"]
  inventory = data["inventory"]
  version = data["version"]
  reward = optimizer.calculate_reward(image_data, reward, inventory)
  next_state_tensor = None
  last_state_tensor = None
  if next_state is not None:
    next_state_tensor = torch.tensor(next_state)
  if last_state is not None:
    last_state_tensor = torch.tensor(last_state)

  memory.push(last_state_tensor, torch.tensor([action]), next_state_tensor, reward)
  optimizer.optimize_model(memory)

  if step % TARGET_UPDATE == 0:
    optimizer.update_target()

  optimizer.save_policy_weights()
  save_step(trial, episode, step, action, reward, last_state, next_state, inventory, version)
  return "step processed"

@app.route('/version', methods=['GET'])
def get_policy_version():
  return str(policy_dqn.version)

@app.route('/policy', methods=['GET'])
def get_policy():
  version = request.args.get('version')
  if(os.path.isfile(os.path.join(STATE_DICT_PATH, 'version_' + version + '_' + optimizer.FILENAME))):
    return make_response(send_from_directory(directory=STATE_DICT_PATH, filename=optimizer.FILENAME, mimetype="application/octet-stream"))
  return make_response("No policy weights have been saved to the server")

def save_step(trial, episode, step, action, reward, last_state, next_state, inventory, version):
  record = {
    "trial": trial,
    "episode": episode,
    "step": step,
    "action": action,
    "reward": reward.numpy()[0].tolist(),
    "last_state": last_state,
    "next_state": next_state,
    "inventory": inventory,
    "version": version
  }

  mongo_db = mongo_client[MONGO_DBNAME]
  mongo_collection = mongo_db["trials"]
  x = mongo_collection.insert_one(record)
  print(x.inserted_id)

if __name__ == '__main__':
  app.config["CACHE_TYPE"] = "null"
  app.run(host=HOST)