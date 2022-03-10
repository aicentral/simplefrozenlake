from pickletools import uint8
import gym
from gym import Env
from gym.spaces import Discrete, Box
import numpy as np
import random
from stable_baselines3 import A2C, DQN, PPO
import cv2
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional

class Drive2D(Env):
    def __init__(self, width, box_size = 100):
        self.width = width
        self.box_size = box_size
        
        # Actions: 0: x+1, 1: y+1
        self.action_space = Discrete(2)
        self.reward_range = (-1, 1)
        self.observation_space = Box(low=np.array([0]*2,dtype=np.float32), high=np.array([self.width]*2,dtype=np.float32))

        self.obstacles = [] # Obstacles or Holes in the map
        for i in range(3):
            self.obstacles.append([i+1,i+2])
        # initial state
        self.state = [0,0]
        self.destination = [self.width-1,self.width-1] 
     
        # Because there is not x-1 or y-1 actions, the max_steps is set to width*2
        # If the charachter can not reach the destination in this number of steps,
        # it will be considered as a failure as there is not much room for corrective movement.
        self.max_steps = self.width*2 

        self.map = np.zeros((self.width*100,self.width*100,3), dtype='uint8') # for rendering
        for ob in self.obstacles:
            self.box_color(ob,[255,255,255]) # render obstacles white
        self.box_color(self.state,[0,0,255]) # render source as red
        
    def mark_destination(self):
        # As the destination will be rendered blue by the charachter when it arrives,
        # we need to mark it green after rendering all the charachter's movements.
        self.box_color(self.destination,[0,255,0])

    def box_color(self, state,color):
        s = 100 #size of box
        self.map[int(state[0])*s:(int(state[0])+1)*s,int(state[1])*s:(int(state[1])+1)*s,:]= color

    def apply_action(self, action):
        # Apply action
        # 0: x+1, 1: y+1
        self.state[action] += 1
        
        if [int(self.state[0]) , int(self.state[1]) ] in self.obstacles:
            return False
        if self.state[0] >= self.width-1 or self.state[1] >= self.width-1:
            return False # out of bounds will end the episode as x-1, y-1 are not valid actions.
        return True
        #else:
        #    self.state[0] = max(0, self.state[0])
        #    self.state[0] = min(self.width, self.state[0])
        #    self.state[1] = max(0, self.state[1])
        #    self.state[1] = min(self.width, self.state[1])

    def distance(self, state1, state2):
        return np.linalg.norm(np.array(state1) - np.array(state2))

    def step(self, action):
        sucess = self.apply_action(action) 
        # if the charachter can not move, it will be considered as a failure
        if not sucess:
            done = True
            reward = -0.1 #-20
        # Check if the charachter is at destination
        if [int(self.state[0]) , int(self.state[1]) ] == self.destination:
            done = True
            reward = 1 #self.width
        else:
            done = False
            reward = 0 #- 0.1*self.distance(self.state, self.destination)/self.width
        
        # Check if max steps have been reached
        if self.max_steps == 0:
            done = True
            reward = 0
        self.max_steps -= 1
        return np.array(self.state,dtype=np.float32), reward, done, {}
        

    def render(self):
        self.box_color(self.state,[255,0,0])
    
    def reset(self):
        self.state = np.array([0,0],dtype=np.float32)
        self.max_steps = self.width*2
        return self.state
        
if __name__ == '__main__':
    LOAD_MODEL_FROM_FILE = True
    WORLD_WIDTH = 8

    env = Drive2D(WORLD_WIDTH)
    #env = gym.make('FrozenLake-v1')
    model = DQN('MlpPolicy', env, verbose=1)#,exploration_initial_eps=0.4,exploration_fraction=0.1,exploration_final_eps=0.1)
    if LOAD_MODEL_FROM_FILE:
        model = DQN.load("dqn_drive2.pkl")
    else:
        model.learn(total_timesteps=400000)
    
    obs = env.reset()
    for i in range(2*WORLD_WIDTH):
        action, _state = model.predict(obs)#, deterministic=True)
        obs, reward, done, info = env.step(int(action))
        print(obs, action,reward, done)
        env.render()
        if done:
            env.mark_destination()
            break # end the loop if the charachter has reached the destination
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    cv2.imwrite('map_{}.png'.format(i), env.map)
    
    if not LOAD_MODEL_FROM_FILE:
        model.save("dqn_drive2.pkl")
