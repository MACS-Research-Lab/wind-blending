import sys
import os
import multiprocessing as mp
from argparse import ArgumentParser, Namespace
import pickle

import optuna
import numpy as np
from rl import learn_rl, evaluate_rl
from stable_baselines3.common.callbacks import BaseCallback
from systems.multirotor import Multirotor, MultirotorTrajEnv, VP

from .opt_pidcontroller import (
    get_controller as get_controller_base,
    apply_params as apply_params_pid,
    make_controller_from_trial,
    make_env,
    DEFAULTS as PID_DEFAULTS
)
from .setup import local_path

DEFAULTS = Namespace(
    ntrials = 100,
    nprocs = 5,
    safety_radius = PID_DEFAULTS.safety_radius,
    max_velocity = PID_DEFAULTS.max_velocity,
    max_acceleration = PID_DEFAULTS.max_acceleration,
    max_tilt = PID_DEFAULTS.max_tilt,
    scurve = PID_DEFAULTS.scurve,
    leashing = False,
    sqrt_scaling = False,
    use_yaw = PID_DEFAULTS.use_yaw,
    wind = PID_DEFAULTS.wind,
    fault = PID_DEFAULTS.fault,
    num_sims = 5,
    max_steps = 50_000,
    study_name = 'MultirotorTrajEnv',
    env_kind = PID_DEFAULTS.env_kind,
    pid_params = '',
    use_trial = None
)

class Callback(BaseCallback):

    def __init__(self, trial: optuna.Trial, verbose: int = 0):
        super().__init__(verbose)
        self.trial = trial
        self.should_stop = False
        self.rollouts = 0

    def _on_rollout_end(self):
        self.rollouts += 1
        self.trial.report(
            value=np.nanmean([info['r'] for info in self.model.ep_info_buffer]),
            step=self.rollouts
        )
        self.should_stop = self.trial.should_prune()

    def _on_step(self) -> bool:
        return not self.should_stop



def get_study(study_name: str=DEFAULTS.study_name, seed:int=0, args: Namespace=DEFAULTS) -> optuna.Study:
    storage_name = "sqlite:///studies/{}.db".format(study_name)
    study = optuna.create_study(
        study_name=study_name, direction='maximize',
        storage=storage_name,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20, # number of rollouts
            n_min_trials=1
        )
    )
    return study



def get_controller(m: Multirotor, trial: optuna.Trial, args: Namespace=DEFAULTS):
    ctrl = get_controller_base(m, scurve=args.scurve, args=args)
    # Taken from PID optimization script, given the default controller params
    if isinstance(args.pid_params, str) and len(args.pid_params) > 0:
        with open(args.pid_params, 'rb') as f:
            params = pickle.load(f)
    elif isinstance(args.pid_params, dict):
        params = args.pid_params
    else:
        params = make_controller_from_trial(trial=trial, args=args, prefix='pid-')
    ctrl.set_params(**params)
    return ctrl

def get_established_controller(m: Multirotor):
    ctrl = get_controller_base(m, scurve=False)

    with open('/home/courseac/projects/transfer_similarity/src/params/manual_pid.pkl', 'rb') as f: #TODO: make relative
        optimal_params = pickle.load(f)

    optimal_params['ctrl_z']['k_p'] = np.array([0.4])
    optimal_params['ctrl_z']['k_i'] = np.array([0.0])
    optimal_params['ctrl_z']['k_d'] = np.array([0.0])
    optimal_params['ctrl_p']['max_velocity'] = 7

    optimal_params['ctrl_a']['k_p'] = np.array([1.5, 2, 2])
    optimal_params['ctrl_a']['k_i'] = np.array([0.5,0,0])
    optimal_params['ctrl_a']['k_d'] = np.array([0.05,0,0])
    optimal_params['ctrl_r']['k_p'] = np.array([10, 10, 10])
    optimal_params['ctrl_r']['k_i'] = np.array([0,0,0])
    optimal_params['ctrl_r']['k_d'] = np.array([0.5,0,0])

    z_params = optimal_params['ctrl_z']
    optimal_params['ctrl_vz']['k_p'] = 25
    vz_params = optimal_params['ctrl_vz']

    ctrl.set_params(**optimal_params)
    ctrl.ctrl_z.set_params(**z_params)
    ctrl.ctrl_vz.set_params(**vz_params)
    return ctrl



def make_objective(args: Namespace=DEFAULTS):
    def objective(trial: optuna.Trial):
        bounding_rect_length = trial.suggest_int('bounding_rect_length', 2, 25) # how long is the bounding rectangle?
        bounding_rect_width = args.safety_corridor
        
        env_kwargs = dict(
            safety_corridor = bounding_rect_width, 
            seed=0,
            get_controller_fn=lambda m: get_established_controller(m, args),
            vp = VP
        )

        
        if args.env_kind=='traj':
            env_kwargs['steps_u'] = trial.suggest_int('steps_u', 1, 50, step=5)
            env_kwargs['scaling_factor'] = trial.suggest_float('scaling_factor', 0.05, 1., step=0.05)
        env = make_env(env_kwargs, args)
        ep_len = env.period // (env.dt * env.steps_u)
        ep_len = ep_len + (32 - (ep_len % 32)) # len as a multiple of 32
        learning_rate = trial.suggest_float('learning_rate', 1e-3, 1e-2, log=True)
        n_epochs = trial.suggest_int('n_epochs', 1, 5)
        n_steps = trial.suggest_int('n_steps', max(ep_len, 32), 10*ep_len, step=ep_len)
        batch_size = trial.suggest_int('batch_size', 32, min(256, n_steps), step=32)
        learn_kwargs = dict(
            steps = args.max_steps,
            n_steps = n_steps,
            learning_rate = learning_rate,
            n_epochs = n_epochs,
            batch_size = batch_size,
            seed=0,
            log_env_params = ('steps_u', 'scaling_factor') if args.env_kind=='traj' else (),
            tensorboard_log = env.name + ('/optstudy/%s/%03d' % (args.study_name, trial.number)),
            policy_kwargs=dict(squash_output=False,
                                net_arch=[dict(pi=[128,128], vf=[128,128])]),
            callback = Callback(trial=trial)
        )
        agent = learn_rl(env, progress_bar=False, **learn_kwargs)
        agent.save(agent.logger.dir + '/agent')
        rew, std, time = evaluate_rl(agent, make_env(env_kwargs, args), args.num_sims) 
        return rew
    return objective



def optimize(args: Namespace=DEFAULTS, seed: int=0, queue=[]):
    study = get_study(args.study_name, seed=seed, args=args)
    for params in queue:
        study.enqueue_trial(params, skip_if_exists=True)
    study.optimize(
        make_objective(args),
        n_trials=args.ntrials//args.nprocs,
    )
    return study



def apply_params(env: MultirotorTrajEnv, **params):
    prefix = 'pid-'
    env_params = ('steps_u', 'scaling_factor')
    rl_params = ('learning_rate', 'n_epochs', 'n_steps', 'batch_size')
    env_dict = {k: params[k] for k in env_params if k in params}
    rl_dict = {k: params[k] for k in rl_params}

    pid_dict = {}
    for k, v in params.items():
        if k in env_params or k in rl_params:
            continue
        elif k.startswith(prefix):
            k = k[len(prefix):]
            pid_dict[k] = v
    apply_params_pid(env.ctrl, **pid_dict)

    env.scaling_factor = env_dict.get('scaling_factor', env.scaling_factor)
    env.steps_u = env_dict.get('steps_u', env.steps_u)
    return dict(rl=rl_dict, pid=pid_dict, env=env_dict)


if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument('study_name', help='Name of study', default=DEFAULTS.study_name, type=str, nargs='?')
    parser.add_argument('--nprocs', help='Number of processes.', default=DEFAULTS.nprocs, type=int)
    parser.add_argument('--ntrials', help='Number of trials.', default=DEFAULTS.ntrials, type=int)
    parser.add_argument('--max_velocity', default=DEFAULTS.max_velocity, type=float)
    parser.add_argument('--max_acceleration', default=DEFAULTS.max_acceleration, type=float)
    parser.add_argument('--max_tilt', default=DEFAULTS.max_tilt, type=float)
    parser.add_argument('--scurve', action='store_true', default=DEFAULTS.scurve)
    parser.add_argument('--leashing', action='store_true', default=DEFAULTS.leashing)
    parser.add_argument('--sqrt_scaling', action='store_true', default=DEFAULTS.sqrt_scaling)
    parser.add_argument('--use_yaw', action='store_true', default=DEFAULTS.use_yaw)
    parser.add_argument('--wind', help='wind force from heading "force@heading"', default=DEFAULTS.wind)
    parser.add_argument('--fault', help='motor loss of effectiveness "loss@motor"', default=DEFAULTS.fault)
    parser.add_argument('--bounding_box', default=DEFAULTS.bounding_box, type=float)
    parser.add_argument('--num_sims', default=DEFAULTS.num_sims, type=int)
    parser.add_argument('--max_steps', default=DEFAULTS.max_steps, type=int)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--pid_params', help='File to load pid params from.', type=str, default=DEFAULTS.pid_params)
    parser.add_argument('--comment', help='Comments to attach to study.', type=str, default='')
    parser.add_argument('--env_kind', help='"[traj,alloc]"', default=DEFAULTS.env_kind)
    parser.add_argument('--use_study', help='Use top 10 parameters from this trial to start', default=None)
    args = parser.parse_args()

    if not args.append:
        import shutil
        path = local_path / ('tensorboard/MultirotorTrajEnv/optstudy/' + args.study_name)
        shutil.rmtree(path=path, ignore_errors=True)
        try:
            os.remove(local_path / ('studies/' + args.study_name + '.db'))
        except OSError:
            pass
    
    # create study if it doesn't exist. The study will be reused with a new seed
    # by each process
    study = get_study(args.study_name, args=args)
    # reuse parameters from another study
    if args.use_study is not None:
        trials = sorted(get_study(args.use_study).trials, key= lambda t: t.value, reverse=True)[:10]
        params = [r.params for r in trials]
        trials_per_proc = len(params) // args.nprocs
        reuse = [params[i:i+trials_per_proc] for i in range(0, len(params), trials_per_proc)]
        remainder = len(params) % trials_per_proc
        if remainder > 0: # add remainder to last process's queue
            reuse[-1].extend(params[-remainder:])
        # for p in params:
        #     study.enqueue_trial(p)
        # reuse = [[] for _ in range(args.nprocs)]
    else:
        reuse = [[] for _ in range(args.nprocs)]

    for key in vars(args):
        study.set_user_attr(key, getattr(args, key))

    mp.set_start_method('spawn')
    with mp.Pool(args.nprocs) as pool:
        pool.starmap(optimize, [(args, i, reuse[i]) for i in range(args.nprocs)])