import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        self.viewer = None
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters # ACTIONS
        self.discrete_action_space = False
        # if true, every agent has the same reward
        self.shared_reward = False
        self.time = 0

        self.world.set_reward_params()

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # physical action space
            action_space = spaces.Box(low=-agent.range, high=+agent.range, shape=(world.act_dim,), dtype=np.float32)
            if agent.acting:
                total_action_space.append(action_space)

            # total action space
            if len(total_action_space) > 1:
                act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            # agent.action.c = np.zeros(self.world.dim_c)

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        # else:
        #     self.viewers = [None] * self.n
        # self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # reset renderer
        # self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # set env action for a particular agent
    def _set_action(self, action, agent, time=None):
        agent.action.turn = action[0]
        agent.action.push = action[1]

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human', close=False):

        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        from gym.envs.classic_control import rendering

        if self.viewer is None:
            self.viewer = rendering.Viewer(self.world.viewer_width, self.world.viewer_height)
        self.viewer.set_bounds(0, self.world.viewer_width/self.world.scale, 0, self.world.viewer_height/self.world.scale)

        # DRAW BACKGROUND
        self.viewer.draw_polygon( [
            (0,                                         0),
            (self.world.viewer_width/self.world.scale,  0),
            (self.world.viewer_width/self.world.scale,  self.world.viewer_height/self.world.scale),
            (0,                                         self.world.viewer_height/self.world.scale),
            ], color=(0., 0., 0.) )

        # DRAW BOUNDARY LINES
        self.viewer.draw_polyline( [
            (self.world.bounds,                    
                self.world.bounds),
            (self.world.viewer_width/self.world.scale - self.world.bounds, 
                self.world.bounds),
            (self.world.viewer_width/self.world.scale - self.world.bounds, 
                self.world.viewer_height/self.world.scale - self.world.bounds),
            (self.world.bounds,                    
                self.world.viewer_height/self.world.scale - self.world.bounds),
            (self.world.bounds,                    
                self.world.bounds),
            ], 
            color=(.75, 0., 0.,),
            linewidth=5)

        bound_color = (0.2, 0.2, 0.2)
        self.viewer.draw_polyline([
            (self.world.border,
                self.world.border),
            (self.world.viewer_width/self.world.scale - self.world.border, 
                self.world.border),
            (self.world.viewer_width/self.world.scale - self.world.border, 
                self.world.viewer_height/self.world.scale - self.world.border),
            (self.world.border,                    
                self.world.viewer_height/self.world.scale - self.world.border),
            (self.world.border,                    
                self.world.border),
            ], 
            color=bound_color,
            linewidth=3)

        self.viewer.draw_polyline( [ 
            (self.world.viewer_width/self.world.scale/3, 
                self.world.border),
            (self.world.viewer_width/self.world.scale/3, 
                self.world.viewer_height/self.world.scale - self.world.border),
            ], 
            color=bound_color,
            linewidth=3)

        self.viewer.draw_polyline( [ 
            (self.world.viewer_width/self.world.scale*2/3, 
                self.world.border),
            (self.world.viewer_width/self.world.scale*2/3, 
                self.world.viewer_height/self.world.scale - self.world.border),
            ], 
            color=bound_color,
            linewidth=3)

        # DRAW FINAL POINTS
        fx, fy = self.world.scale_units(self.world.goal_block.state.pos)
        t = rendering.Transform(translation=(fx, fy))
        self.viewer.draw_circle(
            self.world.scaled_epsilon/self.world.scale, 30, 
            color=self.world.goal_block.goal_color).add_attr(t)

        # DRAW OBJECTS
        for obj in self.world.drawlist:

            if obj.name == "Boundary":
                for w in obj.walls:
                    for f in w.fixtures:
                        trans = f.body.transform
                        path = [trans*v for v in f.shape.vertices]
                        self.viewer.draw_polygon(path, color=obj.color)

            else:
                for f in obj.body.fixtures:
                    trans = f.body.transform
                    path = [trans*v for v in f.shape.vertices]

                    if 'wheel' in f.userData:
                        self.viewer.draw_polygon(path, color=obj.wheel_color)
                    else:
                        self.viewer.draw_polygon(path, color=obj.color)

                # DRAW CP
                if 'agent' in obj.name:
                    x, y = obj.body.worldCenter
                    # print('position: ', obj.position)
                    # print('position: ', obj.worldCenter)
                    t = rendering.Transform(translation=(x, y))
                    self.viewer.draw_circle(0.01, 30, color=obj.pt_color).add_attr(t)

                if 'block' in obj.name:
                    x, y = obj.body.worldCenter
                    # print("world center: ", x, y)
                    # print("local center: ", obj.localCenter)
                    t = rendering.Transform(translation=(x, y))
                    self.viewer.draw_circle(0.01, 30, color=obj.pt_color).add_attr(t)
                    for v in obj.vertices:
                        x, y = obj.body.GetWorldPoint(self.world.scale_units(v))
                        t = rendering.Transform(translation=(x, y))
                        self.viewer.draw_circle(0.005, 30, color=obj.pt_color).add_attr(t)


        return self.viewer.render(return_rgb_array = mode=='rgb_array')



