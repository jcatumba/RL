import os
import copy
import datetime
from types import SimpleNamespace

import numpy as np

from envs import EnvWithGoal
from envs.create_maze_env import create_maze_env
from hiro.models import HiroAgent, TD3Agent
from hiro.utils import Logger, _is_update, record_experience_to_csv


def build_args(**overrides):
    defaults = dict(
        train=False,
        eval=False,
        render=False,
        save_video=False,
        sleep=-1,
        eval_episodes=5,
        env='AntMaze',
        td3=False,
        num_episode=10,
        start_training_steps=100,
        writer_freq=25,
        subgoal_dim=15,
        load_episode=-1,
        model_save_freq=50,
        print_freq=5,
        exp_name=None,
        model_path='model',
        log_path='log',
        policy_freq_low=2,
        policy_freq_high=2,
        buffer_size=50000,
        batch_size=64,
        buffer_freq=10,
        train_freq=10,
        reward_scaling=0.1,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def make_experiment_name(prefix='colab'):
    return prefix + '_' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


def build_env_and_agent(args):
    env = EnvWithGoal(create_maze_env(args.env), args.env)
    goal_dim = 2
    state_dim = env.state_dim
    action_dim = env.action_dim
    scale = env.action_space.high * np.ones(action_dim)

    exp_name = args.exp_name or make_experiment_name('td3' if args.td3 else 'hiro')

    if args.td3:
        agent = TD3Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            scale=scale,
            model_save_freq=args.model_save_freq,
            model_path=os.path.join(args.model_path, exp_name),
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            start_training_steps=args.start_training_steps,
        )
    else:
        agent = HiroAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            goal_dim=goal_dim,
            subgoal_dim=args.subgoal_dim,
            scale_low=scale,
            start_training_steps=args.start_training_steps,
            model_path=os.path.join(args.model_path, exp_name),
            model_save_freq=args.model_save_freq,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            buffer_freq=args.buffer_freq,
            train_freq=args.train_freq,
            reward_scaling=args.reward_scaling,
            policy_freq_high=args.policy_freq_high,
            policy_freq_low=args.policy_freq_low,
        )

    return env, agent, exp_name


class NotebookTrainer:
    def __init__(self, args, env, agent, experiment_name):
        self.args = args
        self.env = env
        self.agent = agent
        log_path = os.path.join(args.log_path, experiment_name)
        self.logger = Logger(log_path=log_path)

    def log(self, global_step, data):
        losses, td_errors = data
        if global_step >= self.args.start_training_steps and _is_update(global_step, self.args.writer_freq):
            for k, v in losses.items():
                self.logger.write('loss/%s' % k, v, global_step)
            for k, v in td_errors.items():
                self.logger.write('td_error/%s' % k, v, global_step)

    def evaluate(self, e):
        if _is_update(e, self.args.print_freq):
            agent = copy.deepcopy(self.agent)
            rewards, success_rate = agent.evaluate_policy(self.env)
            self.logger.write('Success Rate', success_rate, e)
            print(
                'episode:{:05d}, mean:{:.2f}, std:{:.2f}, median:{:.2f}, success:{:.2f}'.format(
                    e, np.mean(rewards), np.std(rewards), np.median(rewards), success_rate
                )
            )

    def train(self):
        global_step = 0

        for e in np.arange(self.args.num_episode) + 1:
            obs = self.env.reset()
            fg = obs['desired_goal']
            s = obs['observation']
            done = False

            step = 0
            episode_reward = 0
            self.agent.set_final_goal(fg)

            while not done:
                a, r, n_s, done = self.agent.step(s, self.env, step, global_step, explore=True)
                self.agent.append(step, s, a, n_s, r, done)
                losses, td_errors = self.agent.train(global_step)
                self.log(global_step, (losses, td_errors))

                s = n_s
                episode_reward += r
                step += 1
                global_step += 1
                self.agent.end_step()

            self.agent.end_episode(e, self.logger)
            self.logger.write('reward/Reward', episode_reward, e)
            self.evaluate(e)


def run_training_experiment(**kwargs):
    args = build_args(train=True, **kwargs)
    env, agent, exp_name = build_env_and_agent(args)
    record_experience_to_csv(args, exp_name)
    trainer = NotebookTrainer(args, env, agent, exp_name)
    trainer.train()
    return {
        'experiment_name': exp_name,
        'model_dir': os.path.join(args.model_path, exp_name),
        'log_dir': os.path.join(args.log_path, exp_name),
    }


def run_evaluation_experiment(exp_name, **kwargs):
    args = build_args(eval=True, exp_name=exp_name, **kwargs)
    env, agent, _ = build_env_and_agent(args)
    agent.load(args.load_episode)
    rewards, success_rate = agent.evaluate_policy(
        env, args.eval_episodes, args.render, args.save_video, args.sleep
    )
    print(
        'mean:{:.2f}, std:{:.2f}, median:{:.2f}, success:{:.2f}'.format(
            np.mean(rewards), np.std(rewards), np.median(rewards), success_rate
        )
    )
    return rewards, success_rate