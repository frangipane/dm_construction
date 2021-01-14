import numpy as np
import argparse
import dm_env
import dm_construction

from helpers import show_rgb_observation


class RandomAgent:
    def __init__(self, ac_spec):
        self.ac_spec = ac_spec

    def act(self, obs):
        action = {}

        for name, spec in self.ac_spec.items():
            if name == "Index":
                # graph observation -> sample edge Index
                #value = np.random.randint(obs["n_edge"])

                # for "smarter" random agent, restrict random choice to available blocks
                moved_block = np.random.randint(7)
                base_block = len(obs["nodes"]) - 1
                value = list(
                    zip(obs["senders"], obs["receivers"])).index((moved_block, base_block))
            elif spec.dtype in (np.int32, np.int64, int):
                # discrete action
                value = np.random.randint(spec.minimum, spec.maximum + 1)
            else:
                # continuous action
                value = np.random.uniform(spec.minimum, spec.maximum)
            action[name] = value
        return action


def rollout(task,
            wrapper,
            difficulty,
            seed,
            num_steps=6,
            random_agent=True,):

    # Create the environment
    env = dm_construction.get_environment(
        task,
        wrapper_type=wrapper,
    )

    #print(env.observation_spec())
    """ {
    'nodes': Array(shape=(0, 18), dtype=dtype('float32'), name='nodes_spec'), 
    'edges': Array(shape=(0, 1), dtype=dtype('float32'), name='edges_spec'), 
    'globals': Array(shape=(1, 1), dtype=dtype('float32'), name='globals_spec'), 
    'n_node': Array(shape=(1,), dtype=dtype('int32'), name='n_node_spec'), 
    'n_edge': Array(shape=(1,), dtype=dtype('int32'), name='n_edge_spec'),
    'receivers': Array(shape=(0,), dtype=dtype('int32'), name='receivers_spec'), 
    'senders': Array(shape=(0,), dtype=dtype('int32'), name='senders_spec')}
    """

    #print(env.action_spec())
    """ {
    'Index': Array(shape=(), dtype=dtype('int32'), name=None),
    'x_action': BoundedArray(shape=(), dtype=dtype('int32'), name=None,
        minimum=0, maximum=14), 
    'sticky': BoundedArray(shape=(), dtype=dtype('int32'), name=None, minimum=0, maximum=1)}
    """
    if random_agent is True:
        model = RandomAgent(env.action_spec())
    else:
        raise NotImplementedError

    # start interaction with environment
    np.random.seed(seed)
    timestep = env.reset(difficulty=difficulty)

    # record trajectory, actions, rgb images for analysis
    trajectory = [timestep]
    actions = [None]
    rgb_imgs = [env.core_env.last_time_step.observation["RGB"]]

#    while timestep.step_type != dm_env.StepType.LAST:
    for _ in range(num_steps):
        if timestep.last():
            timestep = env.reset(difficulty=difficulty)
        action = model.act(timestep.observation)
        timestep = env.step(action)
        trajectory.append(timestep)
        actions.append(action)
        rgb_imgs.append(env.core_env.last_time_step.observation["RGB"])

    env.close()
    return trajectory, actions, rgb_imgs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        "-t",
                        help="Name of dm_construction task",
                        type=str,
                        default="covering",
                        choices=dm_construction.ALL_TASKS,)
    parser.add_argument("--wrapper",
                        "-w",
                        help="Name of wrapper to apply to task",
                        type=str,
                        default="discrete_relative",
                        choices=dm_construction.ALL_WRAPPERS,)
    parser.add_argument("--difficulty",
                        type=int,
                        default=0)
    parser.add_argument("--seed",
                        type=int,
                        default=1234)
    args = parser.parse_args()
    rollout(**vars(args))
