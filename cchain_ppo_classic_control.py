# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
import copy
import collections

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    task_seed: int = 1
    """seed for generating the sequence of environment perturbations (the curriculum)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = False
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL-CCHAIN"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 1600000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-3
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = False
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # continual learning
    num_tasks: int = 10
    """The number of sequential tasks (noise levels)"""
    obs_perturb_range: float = 2.0
    """The standard deviation of the Gaussian noise to be added to observations"""
    
    # C-CHAIN specific arguments
    use_cchain: bool = True
    """if toggled, use C-CHAIN regularization"""
    cchain_reg_coef: float = 1.0
    """the coefficient for the C-CHAIN regularization loss"""
    ref_policy_hist_len: int = 10
    """The number of historical policies to store (corresponds to the list of 10 policies)"""
    ref_policy_idx: int = 2
    """Which historical policy to use as the reference (corresponds to his_idx=2)"""
    cchain_adaptive_coef: bool = True
    """if toggled, use the adaptive coefficient for C-CHAIN"""
    cchain_beta: float = 10000.0
    """The target relative scale (beta) between the PPO loss and the regularization loss."""
    cchain_warmup_iters: int = 10
    """Number of iterations to wait in a task before starting to adapt the coefficient."""
    cchain_running_mean_len: int = 50
    """The length of the running mean window for the losses."""
    activation: str = "leaky_relu"
    """The activation function for the neural networks (relu or leaky_relu)"""
    cchain_coef_max_clip: float = 1000.0
    """The maximum value to which the adaptive C-CHAIN coefficient will be clipped."""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations_per_task: int = 0
    """the number of iterations per task (computed in runtime)"""
    num_total_iterations: int = 0
    """the total number of iterations (computed in runtime)"""

# Wrapper for adding noise the observations
class ObservablePerturbationWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.perturbation = np.zeros(env.observation_space.shape)

    def set_perturbation(self, perturbation):
        self.perturbation = perturbation

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs + self.perturbation, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs + self.perturbation, reward, terminated, truncated, info

def make_env(env_id, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # observation noise
        env = ObservablePerturbationWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, activation="leaky_relu"):
        super().__init__()
        
        if activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU(0.1)
        elif activation == "relu":
            self.activation_fn = nn.ReLU()
            
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        action_dim = envs.single_action_space.n
        hidden_dim = 256

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            self.activation_fn,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.activation_fn,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.activation_fn,
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            self.activation_fn,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.activation_fn,
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            self.activation_fn,
            layer_init(nn.Linear(hidden_dim, action_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)

    def get_probs(self, x):
        logits = self.actor(x)
        return Categorical(logits=logits)


def get_perturbations(env_name, num_tasks, perturb_range):
    temp_env = gym.make(env_name)
    obs_shape = temp_env.observation_space.shape
    perturbations = [np.random.normal(0, perturb_range, obs_shape) for _ in range(num_tasks)]
    perturbations[0] = np.zeros(obs_shape) # First task has no noise
    temp_env.close()
    return perturbations


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.num_total_iterations = args.total_timesteps // args.batch_size
    args.num_iterations_per_task = args.num_total_iterations // args.num_tasks
    
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, 
                   entity=args.wandb_entity,
                   sync_tensorboard=True, 
                   config=vars(args), 
                   name=run_name, 
                   monitor_gym=True, 
                   save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs, activation=args.activation).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.use_cchain:
        ref_agent = Agent(envs).to(device)
        policy_history = collections.deque(maxlen=args.ref_policy_hist_len)
        policy_history.append(copy.deepcopy(agent.state_dict())) # Start with the initial agent
        for param in ref_agent.parameters():
            param.requires_grad = False
    
    # get observation noise for each (k) task
    np.random.seed(args.task_seed) # to generate same tasks for different runs
    perturbations = get_perturbations(args.env_id, args.num_tasks, args.obs_perturb_range)
    np.random.seed(args.seed)

    # PPO storage
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    # Continual learning setting, sequential tasks
    for task_idx in range(args.num_tasks):
        # set a new observation noise
        print(f"\n--- Starting Task {task_idx + 1}/{args.num_tasks} ---")
        current_perturbation = perturbations[task_idx]
        envs.call("set_perturbation", current_perturbation)
        writer.add_scalar("charts/task_id", task_idx + 1, global_step)
        
        # adaptive regularization coef
        if args.use_cchain and args.cchain_adaptive_coef:
            running_pg_loss_hist = collections.deque(maxlen=args.cchain_running_mean_len)
            running_reg_loss_hist = collections.deque(maxlen=args.cchain_running_mean_len)
            dynamic_cchain_reg_coef = 1.0
        elif args.use_cchain:
            dynamic_cchain_reg_coef = args.cchain_beta
        else:
            dynamic_cchain_reg_coef = 0.0
        

        # learning loop
        for iteration in range(1, args.num_iterations_per_task + 1):
            global_iteration = task_idx * args.num_iterations_per_task + iteration
            
            if args.anneal_lr:
                frac = 1.0 - (global_iteration - 1.0) / args.num_total_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            iteration_returns = []
            iteration_lengths = []
            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done
                
                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                # LOGGING
                if "_episode" in infos:
                    done_indices = np.where(infos["_episode"])[0]
                    for idx in done_indices:
                        episodic_return = infos["episode"]["r"][idx]
                        episodic_length = infos["episode"]["l"][idx]
                        print(f"global_step={global_step}, task={task_idx+1}, episodic_return={episodic_return}")
                        iteration_returns.append(episodic_return)
                        iteration_lengths.append(episodic_length)
                            
            
            if len(iteration_returns) > 0:
                writer.add_scalar("charts/mean_episodic_return", np.mean(iteration_returns), global_step)
                writer.add_scalar("charts/mean_episodic_length", np.mean(iteration_lengths), global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)


            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            b_reg_inds = np.arange(args.batch_size)
            all_pg_losses = []
            all_p_reg_losses = []
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                np.random.shuffle(b_reg_inds)

                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]
                    reg_mb_inds = b_reg_inds[start:end]

                    if args.use_cchain:
                        hist_idx = -min(len(policy_history), args.ref_policy_idx)
                        ref_state_dict = policy_history[hist_idx]
                        ref_agent.load_state_dict(ref_state_dict)

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    
                    p_reg_loss = torch.tensor(0.0).to(device)
                    # calculate cchain
                    if args.use_cchain:
                        reg_mb_obs = b_obs[reg_mb_inds]
                        current_action_probs = agent.get_probs(reg_mb_obs).probs
                        with torch.no_grad():
                            ref_action_probs = ref_agent.get_probs(reg_mb_obs).probs
                        p_reg_loss = (-ref_action_probs * torch.log(current_action_probs + 1e-8)).sum(-1).mean()

                    all_pg_losses.append(pg_loss.item())
                    if args.use_cchain:
                        all_p_reg_losses.append(p_reg_loss.item())

                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + dynamic_cchain_reg_coef * p_reg_loss
                        
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    
                    
                    if args.use_cchain:
                        policy_history.append(copy.deepcopy(agent.state_dict()))

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            # adaptive coef change
            if args.use_cchain and args.cchain_adaptive_coef:
                mean_pg_loss_this_iter = np.mean(all_pg_losses)
                mean_reg_loss_this_iter = np.mean(all_p_reg_losses) if len(all_p_reg_losses) > 0 else 1.0
                running_pg_loss_hist.append(mean_pg_loss_this_iter)
                running_reg_loss_hist.append(mean_reg_loss_this_iter)
                if iteration > args.cchain_warmup_iters:
                    mean_abs_pg_loss = np.mean(np.abs(np.array(running_pg_loss_hist)))
                    mean_reg_loss = np.mean(np.array(running_reg_loss_hist))
                    new_coef = args.cchain_beta * mean_abs_pg_loss / (mean_reg_loss + 1e-8)
                    dynamic_cchain_reg_coef = np.clip(new_coef, 1.0, args.cchain_coef_max_clip) 
                    # clipped to prevent exploading coef problem
                    # TODO: results without clipping the coef




            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            if args.use_cchain:
                writer.add_scalar("losses/p_reg_loss", p_reg_loss.item(), global_step)
                writer.add_scalar("charts/reg_coef", dynamic_cchain_reg_coef, global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()