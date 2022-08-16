from multiagent.my_environment import MultiAgentEnv
import multiagent.scenarios as scenarios

if __name__ == '__main__':

    # load scenario from script
    scenario_fn = "simple_blocks.py"
    scenario = scenarios.load(scenario_fn).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer = False)
    env.render()
    obs_n = env.reset()
    
    while True:
        # randomly sample action for each agent
        act_n = []
        for i, act_space in enumerate(env.action_space):
            act_n.append(act_space.sample())
        
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        env.render()
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))