import numpy as np
import Box2D
from Box2D.b2 import (polygonShape, circleShape, staticBody, dynamicBody, vec2, fixtureDef, contactListener, dot)
# from my_core import Agent, Walls, TBlock, ContactDetector 
from my_gym.maenvs.multiagent.my_core import Agent, TBlock, Boundary, ContactDetector


###### HELPER FUNCTION #######################################################################

def distance(pt1, pt2):
    x, y = [(a-b)**2 for a, b in zip(pt1, pt2)]
    return (x+y)**0.5

def unit_vector(bodyA, bodyB):
    Ax, Ay = bodyA.state.pos
    Bx, By = bodyB.state.pos
    denom = max(abs(Bx-Ax), abs(By-Ay))
    return ((Bx-Ax)/denom, (By-Ay)/denom)


###### WORLD CLASS #######################################################################

class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.world = Box2D.b2World(gravity=(0,0), doSleep=False)

        self.agents         = []
        self.blocks         = []
        self.block_queue    = []
        self.boundary       = None
        self.goal_block     = None

        self.bckgnd_color   = (0., 0., 0.)
        self.viewer_width   = int(1440/4)
        self.viewer_height  = int(810/4) 
        self.border         = 0.3   # size of border around which to place objects
        self.bounds         = 0.1   # classifies out of bounds
        self.fps            = 50    # frames per second
        self.scale          = 140   # affects how fast-paced game is and visual scale
        self.soft_force     = True
        self.make_walls     = True

        self.epsilon        = 0.1
        self.shaping        = True
        self.scaled_epsilon = self.epsilon

        self.drawlist       = None
        self.act_dim        = 2

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.blocks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return self.agents

    def norm_units(self, pt):
        ratio = self.scale / self.viewer_width
        return pt[0]*ratio, pt[1]*ratio

    def scale_units(self, pt):
        ratio = self.viewer_width / self.scale
        return pt[0]*ratio, pt[1]*ratio

    def norm_angle(self, a):
        theta = a % (2*np.pi)
        if theta <= np.pi: 
            norm_theta = -theta/np.pi
        else:
            norm_theta = (2*np.pi - theta)/np.pi
        return norm_theta

    def initialize_agents(self):
        for agent in self.agents:
            x = np.random.uniform(self.border, self.viewer_width/self.scale/3-self.border)
            y = np.random.uniform(self.border, self.viewer_height/self.scale-self.border)
            agent.create(self.world, x, y)

    def initialize_blocks(self):
        for block in self.blocks:
            x = np.random.uniform(self.border, self.viewer_width/self.scale/3-self.border)
            y = np.random.uniform(self.border, self.viewer_height/self.scale-self.border)
            block.create(self.world, x, y)
            self.block_queue.append(block.name)

    def initialize_boundary(self):
        if self.boundary: #self.boundary.create(self.world)
            walls = []
            borders = [(0, 1/2), (1, 1/2), (1/2, 0), (1/2, 1)]

            for i, border in enumerate(borders):
                if i < 2:
                    box_shape = [self.bounds, self.viewer_height/self.scale]
                else:
                    box_shape = [self.viewer_width/self.scale, self.bounds]

                wall = self.world.CreateStaticBody(
                    position    = (
                        self.viewer_width  / self.scale * border[0], 
                        self.viewer_height / self.scale * border[1]),
                    fixtures    = fixtureDef(
                        shape   = polygonShape(box=(box_shape)),
                        ),
                    userData = 'wall'
                    )
                walls.append(wall)
            self.boundary.create(walls)

    def reset_contact_listener(self):
        contactListener_bug_workaround = ContactDetector(self.world)
        self.contact_listener = contactListener_bug_workaround

    def set_goal_block(self):
        for block in self.blocks:
            if block.name == self.block_queue[0]:
                self.block_queue.pop(0)
                self.goal_block = block

    def set_random_goal(self):
    
        x = np.random.uniform(
            self.viewer_width / self.scale * 2/3 + self.border, 
            self.viewer_width / self.scale - self.border)
        y = np.random.uniform(
            self.border, 
            self.viewer_height / self.scale - self.border)
        # Normalize units
        x, y = self.norm_units((x,y))
        # set goal
        self.goal_block.set_goal(x, y, 0)

    def out_of_bounds(self, entities):
        for obj in entities:
            x, y = obj.body.worldCenter
            if x < self.bounds or x > (self.viewer_width/self.scale - self.bounds):
                return True
            elif y < self.bounds or y > (self.viewer_height/self.scale - self.bounds):
                return True
        return False

    def update_states(self):
        for block in self.blocks:
            x, y            = self.norm_units(block.body.worldCenter)
            block.state.pos = (x, y)
            block.state.rot = self.norm_angle(block.body.angle)
            if block.goal:
                Fx, Fy, Fangle = block.goal
                block.state.rel_pos = (x-Fx, y-Fy)
                if block.state.dist: block.state.prev_dist = block.state.dist
                block.state.dist = distance((x, y), (Fx, Fy))
            block.state.in_place = block.in_place(self.scaled_epsilon)

        for i, agent in enumerate(self.agents):
            # global inforamtion
            x, y                = self.norm_units(agent.body.worldCenter)
            agent.state.pos     = (x, y)
            agent.state.rot     = self.norm_angle(agent.body.angle)
            agent.state.lin_vel = agent.body.linearVelocity
            agent.state.ang_vel = agent.body.angularVelocity

            # relative distance to block
            Gx, Gy = self.goal_block.state.pos
            agent.state.rel_pos = (x-Gx, y-Gy)
            if agent.state.dist: agent.state.prev_dist = agent.state.dist
            agent.state.dist = distance((x, y), (Gx, Gy))

            # relative distance to vertices
            agent.state.rel_vert_pos = []
            for v in self.goal_block.vertices:
                Vx, Vy = self.norm_units(self.goal_block.body.GetWorldPoint(v))
                agent.state.rel_vert_pos.append((x-Vx, y-Vy))

            # relative distance to other agents 
            agent.state.rel_agent_pos = []
            for o in [a for j, a in enumerate(self.agents) if i != j]:
                Ax, Ay = self.norm_units(o.body.worldCenter)
                agent.state.rel_agent_pos.append((x-Ax, y-Ay))

    def set_reward_params(self, agentDelta=50, agentDistance=0.0, blockDelta=25, blockDistance=0.0,
        puzzleComp=10000, outOfBounds=1000, blkOutOfBounds=100):

        self.weight_deltaAgent          = agentDelta
        self.weight_agent_dist          = agentDistance
        self.weight_deltaBlock          = blockDelta
        self.weight_blk_dist            = blockDistance
        self.puzzle_complete_reward     = puzzleComp
        self.out_of_bounds_penalty      = outOfBounds
        self.blk_out_of_bounds_penalty  = blkOutOfBounds

    @property
    def delta_agent(self):
        return self.weight_deltaAgent

    @property
    def dist_agent(self):
        return self.weight_agent_dist

    @property
    def delta_block(self):
        return self.weight_deltaBlock

    @property
    def dist_block(self):
        return self.weight_blk_dist

    def destroy(self):
        if not self.blocks: return # if NONE then skip reset
        self.world.contactListener = None
        for block in self.blocks:
            self.world.DestroyBody(block.body)
        self.blocks = []
        self.block_queue = []
        for agent in self.agents:
            self.world.DestroyBody(agent.body)
        self.agents = []

    # update state of the world
    def step(self):
        # take agent step??
        for agent in self.agents:
            self.apply_agent_force(agent)
            if self.soft_force: self.apply_soft_force(agent)

        self.world.Step(1.0/self.fps, 6*30, 2*30)
        self.update_states()


    def apply_agent_force(self, agent):
        f = agent._body.GetWorldVector(localVector=(0.0, 1.0))
        p = agent._body.GetWorldPoint(localPoint=(0.0, 0.0)) # change apply point? prev. (0.0, 2.0)
        
        f = ( f[0] * agent.action.push * agent.max_force, 
              f[1] * agent.action.push * agent.max_force )

        agent._body.ApplyForce(f, p, True)
        agent.update_friction()
        agent._body.ApplyAngularImpulse( 0.1 * agent._body.inertia * agent._body.angularVelocity, True )

        # APPLY turn
        torque = abs(agent.action.turn)*agent.max_torque
        # print(torque)
        if abs(agent.action.push) < 0.1: agent.action.turn = 0

        if agent.action.turn < 0:
            agent._body.ApplyTorque(torque, True)
            # print(torque, "apply left torque")
        elif agent.action.turn > 0:
            agent._body.ApplyTorque(-torque, True)
            # print(torque, "apply right torque")
        else:
            agent._body.ApplyTorque(0, True)

    def apply_soft_force(self, agent):
        # APPLY soft force
        force = 10**(-distance(agent.state.pos, self.goal_block.state.pos)) # CHANGE STRENGTH of soft force over time
        force /= 50
        soft_vect = unit_vector(agent, self.goal_block)
        soft_force = (force*soft_vect[0], force*soft_vect[1])
        self.goal_block.body.ApplyForce(soft_force, self.goal_block.state.pos, True)

