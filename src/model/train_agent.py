from dm_control import suite
import tree
from .wrapper_setup import NormilizeActionSpecWrapper
from .wrapper_setup import MujocoActionNormalizer
from .sac import SAC
import torch
from .model_utils import init_hook_dict
from .model_utils import recordtodict_hook
from .model_utils import get_flat_obs
from .model_utils import get_kinematics
from .model_utils import compile_hook_dict
from acme import wrappers
import numpy as np


def extract_kinematic_activations():
    # load the environment
    env = suite.load(domain_name="cheetah", task_name="run")
    # add wrappers onto the environment
    env = NormilizeActionSpecWrapper(env)
    env = MujocoActionNormalizer(environment=env, rescale='clip')
    env = wrappers.SinglePrecisionWrapper(env)


    class Args:
        env_name = 'whatever'
        policy = 'Gaussian'
        eval = True
        gamma = 0.99
        tau = 0.005
        lr = 0.0003
        alpha = 0.2
        automatic_entropy_tuning = True
        seed = 42
        batch_size = 256
        num_steps = 1000000
        hidden_size = 256
        updates_per_step = 1
        start_steps = 10000
        target_update_interval = 1
        replay_size = 1000000
        cuda = False


    args = Args()

    # get the dimensionality of the observation_spec after flattening
    flat_obs = tree.flatten(env.observation_spec())
    # combine all the shapes
    obs_dim = sum([item.shape[0] for item in flat_obs])

    # setup agent
    agent = SAC(obs_dim, env.action_spec(), args)

    # load checkpoint - UPLOAD YOUR FILE HERE!
    model_path = '/content/sac_checkpoint_cheetah_123456_10000'
    agent.load_checkpoint(model_path, evaluate=True)

    # pull out model
    model = agent.policy
    # setup hook dict
    hook_dict = init_hook_dict(model)
    # add hooks
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module.register_forward_hook(recordtodict_hook(name=name, hook_dict=hook_dict))

    #collect activations and kinematics
    CHEETAH_GEOM_NAMES = ['ground', 'torso', 'head', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
    CHEETAH_JOINT_NAMES = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
    CHEETAH_ACTUATOR_NAMES = ['bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']

    # get the mapping of the geom names
    geom_names_to_idx = {geom_name: idx for idx, geom_name in enumerate(CHEETAH_GEOM_NAMES)}
    # get the mapping of the joint names
    joint_names_to_idx = {joint_name: idx for idx, joint_name in enumerate(CHEETAH_JOINT_NAMES)}
    # get the mapping of the actuator names
    actuator_names_to_idx = {actuator_name: idx for idx, actuator_name in enumerate(CHEETAH_ACTUATOR_NAMES)}


    # run a few episodes just to collect activations
    num_episodes_to_run = 42

    # for recording kinematics
    total_kinematic_dict = {
        'geom_positions': [],
        'joint_angles': [],
        'joint_velocities': [],
        'actuator_forces': []
    }

    for i in range(num_episodes_to_run):
        time_step = env.reset()
        episode_reward = 0
        while not time_step.last():  # or env.get_termination()
            # get the state
            state = get_flat_obs(time_step)
            # sample an action
            action = agent.select_action(state)
            time_step = env.step(action)

            # record kinematics
            kinematic_dict = get_kinematics(env.physics, CHEETAH_GEOM_NAMES, CHEETAH_JOINT_NAMES, CHEETAH_ACTUATOR_NAMES)
            total_kinematic_dict['geom_positions'].append(kinematic_dict['geom_positions'])
            total_kinematic_dict['joint_angles'].append(kinematic_dict['joint_angles'])
            total_kinematic_dict['joint_velocities'].append(kinematic_dict['joint_velocities'])
            total_kinematic_dict['actuator_forces'].append(kinematic_dict['actuator_forces'])
            # record reward
            episode_reward += time_step.reward
        print('Episode: {} Reward: {}'.format(i, episode_reward))


    loaded_hook_dict = compile_hook_dict(hook_dict)

    # process the kinematics - convert the kinematics to numpy arrays
    total_kinematic_dict['geom_positions'] = np.stack(total_kinematic_dict['geom_positions'],
                                                    axis=0)  # combine the geom_positions_arr into (t, n, 3)
    total_kinematic_dict['joint_angles'] = np.array(total_kinematic_dict['joint_angles'])
    total_kinematic_dict['joint_velocities'] = np.array(total_kinematic_dict['joint_velocities'])
    total_kinematic_dict['actuator_forces'] = np.array(total_kinematic_dict['actuator_forces'])

    return loaded_hook_dict, total_kinematic_dict