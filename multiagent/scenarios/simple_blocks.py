import numpy as np
from multiagent.scenario import BaseScenario
from my_gym.maenvs.multiagent.my_world import World
from my_gym.maenvs.multiagent.my_core import Agent, TBlock, Boundary

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

    def reward(self, agent, world):
        reward = 0

        # DELTA / DISTANCE of block 
        # reward += (world.goal_block.state.prev_dist - world.goal_block.state.dist) * world.delta_block
        # # print("block deltadistance: %s" % (deltaDist*200)) # <0.05
        reward -= world.dist_block * world.goal_block.state.dist

        # Delta Agent
        # reward += (agent.state.prev_dist - agent.state.dist) * world.delta_agent
        # Distance Agent
        reward -= world.dist_agent * agent.state.dist

        return reward
               
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

        return np.array(state)


    def done(self, agent, world):
        # if block in place -- done
        if world.goal_block.state._in_place:
            return True
        return False


