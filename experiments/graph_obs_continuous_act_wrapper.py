"""Environment wrapper (meta) returning graph observations from
discrete_relative wrapper and implementing continuous absolute actions
via continuous_absolute wrapper.

Compatible with MLPGaussianActorCritic model's inputs/outputs.

observation_spec() ==>
    {
    'nodes': Array(shape=(0, 18), dtype=dtype('float32'), name='nodes_spec'),
    'edges': Array(shape=(0, 1), dtype=dtype('float32'), name='edges_spec'),
    'globals': Array(shape=(1, 1), dtype=dtype('float32'), name='globals_spec'),
    'n_node': Array(shape=(1,), dtype=dtype('int32'), name='n_node_spec'),
    'n_edge': Array(shape=(1,), dtype=dtype('int32'), name='n_edge_spec'),
    'receivers': Array(shape=(0,), dtype=dtype('int32'), name='receivers_spec'),
    'senders': Array(shape=(0,), dtype=dtype('int32'), name='senders_spec')
    }


action_spec() ==>
    {
    'Horizontal': BoundedArray(shape=(), dtype=dtype('float32'), name=None, minimum=-7.5, maximum=7.5),
    'Vertical': BoundedArray(shape=(), dtype=dtype('float32'), name=None, minimum=0.0, maximum=15.0),
    'Sticky': BoundedArray(shape=(), dtype=dtype('float32'), name=None, minimum=-1.0, maximum=1.0),
    'Selector': BoundedArray(shape=(), dtype=dtype('float32'), name=None, minimum=-7.0, maximum=7.0)
    }

"""
import numpy as np

import dm_construction
from dm_construction.wrappers.discrete_relative import DiscreteRelativeGraphWrapper

import model


class ContinuousAbsoluteGraphWrapper(DiscreteRelativeGraphWrapper):
    """
    Graph observations (from discrete_relative wrapper),
    continuous absolute actions (continuous_absolute wrapper)
    """
    def __init__(self, env, **discrete_relative_kwargs):
        super().__init__(env, **discrete_relative_kwargs)
        self.ca_env = dm_construction.get_wrapped_environment(self._env, "continuous_absolute")

    def step(self, action):
        """continuous absolute action"""
        _ = self.ca_env.step(action)

        # get updated graph obs
        timestep = self._process_time_step(self._env.last_time_step)
        return timestep

    def action_spec(self):
        """continuous absolute action"""
        return self.ca_env.action_spec()


def mixed_wrapper_rollout(task,
                          difficulty,
                          seed,
                          num_steps=6,
                          model_constructor=model.ActorCritic
                          ):
    """
    Rollout with graph observations but continuous actions, with
    observations coming from `discrete_relative` wrapper, and actions
    invoked through `continuous_absolute` wrapper.
    """
    # Create the environment
    unity_env = dm_construction.get_unity_environment(backend="docker")
    task_env = dm_construction.get_task_environment(unity_env, problem_type=task,)
    env = ContinuousAbsoluteGraphWrapper(task_env)

    # Create model
    model = model_constructor(ob_spec=env.observation_spec(),
                              ac_spec=env.action_spec())

    # start interaction with environment
    np.random.seed(seed)
    timestep = env.reset(difficulty=difficulty)

    # record trajectory, actions, rgb images for analysis
    trajectory = [timestep]
    actions = [None]
    rgb_imgs = [task_env.last_time_step.observation["RGB"]]

#    while timestep.step_type != dm_env.StepType.LAST:
    for _ in range(num_steps):
        if timestep.last():
            timestep = env.reset(difficulty=difficulty)

        action = model.act(timestep.observation)

        # Take action via continous_absolute wrapper
        timestep = env.step(action)

        # Update record
        trajectory.append(timestep)
        actions.append(action)
        rgb_imgs.append(task_env.last_time_step.observation["RGB"])

    env.close()
    return trajectory, actions, rgb_imgs


if __name__ == "__main__":
    mixed_wrapper_rollout("covering", 0, 1234, 6)
