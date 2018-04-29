import numpy as np
# from multiagent.core import World, Agent, Landmark
from multiagent.scenario import BaseScenario
from my_gym.maenvs.multiagent.my_world import World
from my_gym.maenvs.multiagent.my_core import Agent, TBlock, Boundary

# from my_world import World
# from my_core import Agent, TBlock, Boundary
import random

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        
        # set any world properties first
        self.num_agents = 2

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.destroy()
        world.reset_contact_listener()

        # add agents
        world.agents = [Agent() for i in range(self.num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent_%d' % i

        # add blocks
        world.blocks = [TBlock()]

        if world.make_walls: 
            world.boundary = Boundary()
            world.initialize_boundary()

        # initialize blocks + agents in simulation
        world.initialize_agents()
        world.initialize_blocks()

        world.set_goal_block()
        world.set_random_goal()
        world.update_states()

        # self.done_status = None

        world.drawlist = [world.boundary] + world.blocks + world.agents

 #       # # random properties for landmarks
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.color = np.array([0.1, 0.1, 0.1])
        #     landmark.color[i + 1] += 0.8
        #     landmark.index = i
        # # set goal landmark
        # goal = np.random.choice(world.landmarks)
        # for i, agent in enumerate(world.agents):
        #     agent.goal_a = goal
        #     agent.color = np.array([0.25, 0.25, 0.25])
        #     if agent.adversary:
        #         agent.color = np.array([0.75, 0.25, 0.25])
        #     else:
        #         j = goal.index
        #         agent.color[j + 1] += 0.5
        # # set random initial states
        # for agent in world.agents:
        #     agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     agent.state.p_vel = np.zeros(world.dim_p)
        #     agent.state.c = np.zeros(world.dim_c)
        # for i, landmark in enumerate(world.landmarks):
        #     landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
        #     landmark.state.p_vel = np.zeros(world.dim_p)

    def reward(self, agent, world):
        reward = 0

        # DELTA / DISTANCE of block 
        reward += (world.goal_block.state.prev_dist - world.goal_block.state.dist) * world.delta_block
        # # print("block deltadistance: %s" % (deltaDist*200)) # <0.05
        reward -= world.dist_block * world.goal_block.state.dist

        # Delta Agent
        reward += (agent.state.prev_dist - agent.state.dist) * world.delta_agent
        # Distance Agent
        reward -= world.dist_agent * agent.state.dist

        return reward

    # def agent_reward(self, agent, world):
    #     # the distance to the goal
    #     return -np.sqrt(np.sum(np.square(agent.state.p_pos - agent.goal_a.state.p_pos)))
               
    def observation(self, agent, world):
        state = [
            agent.state.rot,                    # agent rotation
            agent.state.lin_vel[0],             # linear velocity
            agent.state.lin_vel[1],
            agent.state.ang_vel,                # angular velocity
            agent.state.rel_pos[0],             # rel position to block centroid
            agent.state.rel_pos[1],
            agent.state.dist,
            1.0 if agent.goal_contact else 0.0, # in contact with block
            ]

        # relative position of agent to goal_block vertices
        for v in agent.state.rel_vert_pos:
            state.extend([v[0], v[1]])

        # relative position to other agents
        for a in agent.state.rel_agent_pos:
            state.extend([a[0], a[1]])

        # block relative location
        state.extend([
            world.goal_block.state.rel_pos[0],
            world.goal_block.state.rel_pos[1],
            world.goal_block.state.dist,
            world.goal_block.state.rot,
            1.0 if world.goal_block.state._in_place else 0.0,
            ])

        # print(agent.name, state)

        return np.array(state)


    def done(self, agent, world):
        # if block in place -- done
        # implement boundary later?
        # print("CHECKING IF DONE")
        # print("goal block return: ", world.goal_block.state._in_place)
        # if world.goal_block.in_place(world.epsilon/world.scale):
        if world.goal_block.state._in_place:
            print("IN PLACE")
            return True
        return False


