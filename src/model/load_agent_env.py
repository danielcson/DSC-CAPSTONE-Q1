from acme import wrappers
from dm_control import suite
import tree
from wrappers import NormilizeActionSpecWrapper
from wrappers import MujocoActionNormalizer
from sac import SAC
from utils import init_hook_dict
from utils import recordtodict_hook
import torch

def load_agent():
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