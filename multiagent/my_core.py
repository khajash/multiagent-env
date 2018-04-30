import numpy as np
import Box2D
from Box2D.b2 import (polygonShape, circleShape, staticBody, dynamicBody, vec2, fixtureDef, contactListener, dot)


###### CONTACT CLASS #######################################################################

class ContactDetector(contactListener):
    def __init__(self, world):
        contactListener.__init__(self)
        self.world = world
    def BeginContact(self, contact):
        # if block and agent in touch
        for agent in self.world.agents:
            if agent.body in [contact.fixtureA.body, contact.fixtureB.body]:
                if self.world.goal_block.body in [contact.fixtureA.body, contact.fixtureB.body]:
                    agent.goal_contact = True

    def EndContact(self, contact):
        for agent in self.world.agents:
            if agent.body in [contact.fixtureA.body, contact.fixtureB.body]:
                if self.world.goal_block.body in [contact.fixtureA.body, contact.fixtureB.body]:
                    agent.goal_contact = False

###### STATE CLASSES #######################################################################

# # physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.pos = None
        # physical rotation
        self.rot = None
        # relative distance to goal
        self.rel_pos = None
        self.dist = None
        self.prev_dist = None
    
# # state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # relative distance to vertices of block
        self.rel_vert_pos = []
        # relative distance to other agents
        self.rel_agent_pos = []
        # physical linear velocity
        self.lin_vel = None
        # physical linear velocity
        self.ang_vel = None

class BlockState(EntityState):
    def __init__(self):
        super(BlockState, self).__init__()
        self._in_place = False

###### ACTION CLASS #######################################################################

# # action of the agent
class Action(object):
    def __init__(self):
        self.turn = None
        self.push = None

###### ENTITY CLASSES #######################################################################

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # body - to contain main Box2D body
        self._body = None
        self._state = EntityState() 
        # name 
        self.name = ''
        # entity can move / be pushed
        self.dynamic = True
        # completes actions
        self.acts = False
        self.action = None
        # material density (affects mass)
        self.density = 1.0
        # restitution of object (0. = no bounce)
        self.restitution = 0.0
        # friction between objects
        self.friction = 0.01
        # linear damping (affects lin. movement through space)
        self.linear_damping = 5.0
        # angular damping (affects rotation through space)
        self.angular_damping = 5.0
        # color (body + points)
        self.color = None
        self.pt_color = None

    @property
    def body(self):
        return self._body

    @property
    def acting(self):
        return self.acts

    @property
    def state(self):
        return self._state



# properties of landmark entities
class Block(Entity):
    def __init__(self):
        super(Block, self).__init__()
        self.density    = 1.56 # measured density of foam
        self.color      = (0.5, 0.5, 0.5)
        self.pt_color   = (1., 1., 1.)
        self.goal_color = (58./255, 153./255, 1)
        self._vertices  = []
        self._goal      = None
        self._state     = BlockState()
        self.friction   = 0.15

    def create(self, world, x, y):
        pass

    def set_goal(self, x, y, theta=0):
        self._goal = (x, y, theta)

    def in_place(self, epsilon):
        f_x, f_y, f_angle   = self._goal
        x, y                = self._state.pos

        if abs(f_x - x) > epsilon:
            self._in_place = False
            return False
        if abs(f_y - y) > epsilon:
            self._in_place = False
            return False
        self._in_place = True
        return True

    @property
    def vertices(self):
        return self._vertices

    def scale_vertices(self, scale):
        self._vertices = [scale(v) for v in self._vertices]

    @property
    def goal(self):
        return self._goal


class TBlock(Block):
    def __init__(self):
        super(TBlock, self).__init__()
        self.name = 'T_block'

    def create(self, world, x, y):
        self._body = world.CreateDynamicBody(
            position        = (x, y),
            fixtures        = [
                fixtureDef(
                    shape       = polygonShape(box = (0.1, 0.1, (0., -0.1),0)),
                    density     = self.density, 
                    friction    = self.friction, 
                    restitution = self.restitution,
                    userData    = 'short block'),
                fixtureDef(
                    shape       = polygonShape(box = (0.3, 0.1, (0., 0.1),0)),
                    density     = self.density, 
                    friction    = self.friction, 
                    restitution = self.restitution,
                    userData    = 'long block'),
                ],
            angle           = np.random.uniform(0, 2*np.pi), 
            linearDamping   = self.linear_damping, 
            angularDamping  = self.angular_damping,
            userData        = self.name
            )

        self._body.agent_contact = False # store for contact

        for fix in self.body.fixtures:
            if self._vertices:
                extend_v = [v for v in fix.shape.vertices if v not in self._vertices]
                self._vertices.extend(extend_v)
            else:
                self._vertices = fix.shape.vertices

class Boundary(Entity):
    def __init__(self):
        super(Boundary, self).__init__()
        self.dynamic    = False
        self.color      = (0.2, 0.2, 0.2)
        self.walls      = []
        self.name       = "Boundary"

    def create(self, walls):
        self.walls = walls

# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.dynamic      = True
        self.acts         = True
        self.density      = 17.3
        self._max_force   = 1.0
        self._max_torque  = 0.001
        self.goal_contact = False

        # action
        self.action = Action()
        # control range
        self.range        = 1.0
        self._state = AgentState()

        self.hull = [
            (-0.039,-0.095), (0.039,-0.095), (0.095,-0.039), (0.095,0.039), 
            (0.039,0.095), (-0.039,0.095), (-0.095,0.039), (-0.095,-0.039)
            ]
        self.color          = (1., 1., 1.)
        self.wheel_color    = (0.5, 0.5, 0.5)
        self.pt_color       = (0.5, 0.5, 0.5)

    @property
    def max_force(self):
        return self._max_force

    @property
    def max_torque(self):
        return self._max_torque

    def create(self, world, x, y, theta=3/2*np.pi):
        self._body = world.CreateDynamicBody(
                position = (x, y),
                fixtures = [
                    fixtureDef(
                        shape           = polygonShape(vertices=[(x,y) for x,y in self.hull]),
                        density         = self.density,
                        friction        = self.friction,
                        restitution     = self.restitution,
                        userData        = 'body'),
                    fixtureDef(
                        shape           = polygonShape(box=(0.005, 0.05, (0.06, 0.),0)), 
                        density         = 0, 
                        friction        = self.friction, 
                        restitution     = self.restitution,
                        userData        = 'wheel1'),
                    fixtureDef(
                        shape           = polygonShape(box=(0.005, 0.05, (-0.06, 0.),0)),
                        density         = 0, 
                        friction        = self.friction, 
                        restitution     = self.restitution,
                        userData        = 'wheel2'),
                    ],
                angle           = theta, 
                linearDamping   = self.linear_damping, 
                angularDamping  = self.angular_damping,
                userData        = self.name,
                )

        

    def get_lateral_velocity(self):
        currentRightNormal = self.body.GetWorldVector(localVector=(1.0, 0.0))
        return dot(currentRightNormal, self.body.linearVelocity) * currentRightNormal

    def update_friction(self):
        impulse = self.body.mass * -self.get_lateral_velocity() 
        self.body.ApplyLinearImpulse(impulse, self.body.worldCenter, True)

