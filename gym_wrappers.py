import gym
import numpy as np
import keras
from image_functions import preprocess_image as pme_prepro

class PlatformermanEyesWrapper(gym.ObservationWrapper):
  """
  NOTE: This wrapper will not work if other preprocessing wrappers are called
  before it
  """
  def __init__(self, env, modelPath = "./PMEModel.h5", numLabels = 5):
    super(PlatformermanEyesWrapper, self).__init__(env)
    self.PMModel = keras.models.load_model(modelPath)
    modelOutShape = self.PMModel.layers[-1].output.shape
    boxShape = (1, modelOutShape[1], modelOutShape[2])

    self.numLabels = numLabels

    print("SEG WRAPPER: Initialising segmentation wrapper with shape:", boxShape)
    print("SEG WRAPPER: number of labels", self.numLabels)

    self.observation_space = gym.spaces.Box(
      low = 0.0,
      high = float(self.numLabels),
      shape = boxShape,
      dtype = np.float32
    )

  def observation(self, obs):
    pred = self.PMModel.predict(pme_prepro(obs)[np.newaxis,...],
      verbose = 0)[...]
    pred = np.round(pred)
    pred = pred.clip(min = 0, max = self.numLabels - 1)
    pred = pred/self.numLabels
    return pred

class NESActionDiscretizer(gym.ActionWrapper):
  """
  converts the environment actions from multibinary to discrete.
  Adapted from the SonicDiscretizer here: 
  https://github.com/openai/baselines/blob/master/baselines/common/retro_wrappers.py
  """
  def __init__(self, env):
    super(NESActionDiscretizer, self).__init__(env)
    buttons = \
      ["B", "NONE", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A"]
    actions = [
        ["LEFT"], 
        ["RIGHT"], 
        ["B"], 
        ["A"], 
        ["DOWN"], 
        ["RIGHT", "A"], 
        ["RIGHT", "B"], 
        ["LEFT", "A"], 
        ["LEFT", "B"]]

    self._actions = []
    for action in actions:
      arr = np.array([0] * 9)
      for button in action:
          arr[buttons.index(button)] = 1
      self._actions.append(arr)


    self.action_space = gym.spaces.Discrete(len(self._actions))

  def action(self, a):
    return self._actions[a].copy()

class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(80,80,1), dtype=np.uint8)
    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):

        new_frame = np.reshape(frame, frame.shape).astype(np.float32)

        new_frame = 0.299*new_frame[:,:,0] + 0.587*new_frame[:,:,1] + \
                    0.114*new_frame[:,:,2]

        new_frame = new_frame[35:195:2, ::2].reshape(80,80,1)
        #plt.imshow(new_frame.reshape(80,80))
        #plt.show()

        return new_frame.astype(np.uint8)

class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                            shape=(self.observation_space.shape[-1],
                                   self.observation_space.shape[0],
                                   self.observation_space.shape[1]),
                            dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0

class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                             env.observation_space.low.repeat(n_steps, axis=0),
                             env.observation_space.high.repeat(n_steps, axis=0),
                             dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer