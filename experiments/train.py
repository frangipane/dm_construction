import numpy as np
import argparse
import dm_env
import dm_construction

from helpers import show_rgb_observation


class DiscreteRelativePolicy:
    def __init__(self, ob_spec, ac_spec):
        self.ob_spec = ob_spec
        self.ac_spec = ac_spec

        self.x_action_min = ac_spec['x_action'].minimum
        self.x_action_max = ac_spec['x_action'].maximum

    def act(self, obs):
        pass


class RandomAgent(DiscreteRelativePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def act(self, obs):

        # constrict random choice to available blocks
        moved_block = np.random.randint(7)
        base_block = len(obs["nodes"]) - 1
        edge_index = list(
            zip(obs["senders"], obs["receivers"])).index((moved_block, base_block))

        action = {
            "Index": edge_index,
            "sticky": np.random.choice([0,1]),
            "x_action": np.random.randint(
                self.x_action_min,
                self.x_action_max + 1
            )
        }
        return action


def train(task,
          wrapper,
          difficulty,
          seed,
          random_agent=True,
          ):

    # Create the environment
    env = dm_construction.get_environment(
        task,
        wrapper_type=wrapper,
        difficulty=difficulty,
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
        model = RandomAgent(env.observation_spec(), env.action_spec())
    else:
        raise NotImplementedError

    # start interaction with environment
    np.random.seed(seed)
    timestep = env.reset()

    while timestep.step_type != dm_env.StepType.LAST:
        action = model.act(timestep.observation)
        timestep = env.step(action)
        #show_rgb_observation(env.core_env.last_time_step.observation["RGB"])

    env.close()


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
    train(**vars(args))
