# DQNCloud

A cloud deployment for DQN optimization network. Takes a POST to the `/optimize` endpoint containing previous/next state data, environment rewards, and trial/episode/step information.

## /optimize

POST header should be application/json

POST body should be:

    {
      "image": <list>,
      "reward": <list>,
      "step": <int>,
      "episode": <int>,
      "action": <list>,
      "last_state": <list>,
      "next_state": <list>,
      "inventory": <bool>,
      "version": <int>
    }

## /version

GET the current version of the policy DQN.

## /policy?version=

GET this url to download a copy of the latest policy weights optimized by the network. It takes one query string parameter, `version` which specifies which policy net weights version file to download.

## config.py

A config.py needs to be added to the directory. It should have the following constants defined:

    CONVOLUTION_WIDTH = <int> # How wide will the initial convolution layer be?
    CONVOLUTION_HEIGHT = <int> # How high will the initial convolution layer be?
    BATCH_SIZE = <int> #how large of a batch will be processed by the DQN on optimization
    GAMMA = <float> #The gamma to use in discounting rewards over time
    DEVICE = <string> # pyTorch device (cpu or cuda)
    TARGET_UPDATE = <int> #How many episodes before updating target DQN network
    STATE_DICT_PATH = <string> #Server directory path for storing the state weight dictionary
    REPLAY_MEMORY = <int> #Size of the replay memory
    MONGO_CLIENT = <string> #ip:port of mongoDB instance
    ACTION_SPACES = <list> #Array of integers containing the number of actions for each discreet action space
    HOST = <string> #The IP of the host running this server
