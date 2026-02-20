import torch
import numpy
import pickle
import os
from os.path import join
from tqdm import tqdm
from torch.utils.data import DataLoader

from env.envs import make_env
from agent.ppo import PPO
from agent.ppo_conditional import ConditionalPPO
from common import video
from common.extra import print_message, free_memory, MAE


EVAL_TIMEOUT = 200 # to get a rough idea of performance, spend more runtime on data collection/training
RENDER_EVERY_TEST = True

def train_ppo(cfg, env, net, logger, trial_id, max_rew):
  device = torch.device(cfg["device"])

  ppo_kwargs = dict(
      lr=cfg["lr"],
      lr_a=cfg["lr_a"],
      lr_c=cfg["lr_c"],
      gamma=cfg["gamma"],
      batch_size=cfg["batch_size"],
      gae_lambda=cfg["gae_lambda"],
      clip=cfg["clip"],
      value_coef=cfg["value_coef"],
      entropy_coef=cfg["entropy_coef"],
      epochs_per_step=cfg["epochs_per_step"],
      num_steps=cfg["num_steps"],
      horizon=cfg["horizon"],
      enable_reward_norm=cfg['use_normalised_rewards'],
      input_noise_level=cfg['input_noise_level'],
      device=device
  )
  # set up evaluation env and conditioning type
  test_env, env_args = make_env(cfg)
  z_type = cfg['z_type']
  use_gt_cues = 'ground_truth' in z_type

  if cfg['policy_type'] == 'default' or cfg['z_type'] == 'none':
    ppo = PPO(env, net, **ppo_kwargs)
  else:
    ppo_kwargs['env_args'] = env_args
    ppo_kwargs['use_gt_cues'] = use_gt_cues
    ppo_kwargs['generator_checkpoint_path'] = cfg['generator_checkpoint_path']
    ppo_kwargs['z_noise_level'] = cfg['z_noise_level']
    ppo_kwargs['should_finetune'] = cfg['generator']['should_finetune']
    ppo = ConditionalPPO(env, net, **ppo_kwargs)

  # optionally load agent parameters from checkpoint
  if cfg['checkpoint_path'] is not None:
    #assert cfg['load_epoch'] > 0, "load_epoch must be greater than 0 when checkpoint_path is provided"
    model_path = join(cfg["checkpoint_path"], f"net_{cfg['load_epoch']}.pth")
    print_message(f"loading model from '{model_path}'", mode=0)
    if os.path.exists(model_path):
      ppo.load(model_path)
    else:
      raise Exception(f"can't load '{model_path}'")

  # prepare models for training
  ppo.net.train()
  if cfg['policy_type'] == 'conditional' and not use_gt_cues and ppo.generator.should_finetune: 
    ppo.generator.train()

  icue = env.slice_cues(z_type)
  for step in tqdm(range(cfg['load_epoch'], cfg['num_steps'] + 1)):
    ppo._set_step_params(step)

    #
    # collect experience over one horizon length
    #

    with torch.no_grad():
      memory = ppo.collect_trajectory(cfg["horizon"])

    memory_loader = DataLoader(
      memory, batch_size=cfg["batch_size"], shuffle=True#, num_workers=multiprocessing.cpu_count()
    )

    #
    # train policy over batched experience
    # note: collected experience does not accumulate like replay buffer
    #       agent trains only on most recently experienced horizon
    #
    if cfg['policy_type'] == 'conditional' and not use_gt_cues:
      ppo.generator.create_context() # refresh generator context

    avg_loss = 0.0
    avg_rwd = 0.0
    avg_log_probs = 0.0
    rollout_count = 0.0
    pol, val, ent, zerr = 0, 0, 0, 0
    n_batches = len(memory_loader)
    curr_batch_size = 1
    for epoch in range(cfg["epochs_per_step"]):
      for (states, actions, log_probs, rewards, advantages, values, cues, dones) in memory_loader:
        curr_batch_size = actions.shape[0]
        if isinstance(states, list): # TODO: more elegant way of passing context
          [s.to(device) for s in  states]
        else:
          states.to(device)

        loss, pol_loss, val_loss, ent_loss, z_error = ppo.train_batch(
          states, actions.to(device), log_probs, rewards.to(device), advantages, values, cues
        )
        avg_loss += loss
        avg_rwd += rewards.sum(dim=0).cpu().numpy()
        avg_log_probs += log_probs.sum(dim=0).cpu().numpy()
        rollout_count += dones.count_nonzero()
        pol, val, ent, zerr = pol + pol_loss, val + val_loss, ent + ent_loss, zerr + z_error

    if step == cfg['load_epoch']:
      for name, param in ppo.net.named_parameters():
          if param.grad is None: print(f"frozen: {name}")
          elif param.grad.abs().sum() == 0: print(f"unused: {name}")

    # log training metrics for learning curve
    logger.log("total_loss", avg_loss / n_batches / curr_batch_size / cfg["epochs_per_step"], is_train=True)
    logger.log("pol_loss", pol / n_batches / curr_batch_size / cfg["epochs_per_step"], is_train=True)
    logger.log("val_loss", val / n_batches / curr_batch_size / cfg["epochs_per_step"], is_train=True)
    logger.log("ent_loss", ent / n_batches / curr_batch_size / cfg["epochs_per_step"], is_train=True)
    logger.log("generator_loss", zerr / n_batches / curr_batch_size / cfg["epochs_per_step"], is_train=True)
    logger.log("rewards", avg_rwd / n_batches / curr_batch_size / cfg["epochs_per_step"], is_train=True)
    logger.log("logprobs", avg_log_probs / n_batches / curr_batch_size / cfg["epochs_per_step"], is_train=True)

    should_record_visual_output = RENDER_EVERY_TEST or step == cfg['num_steps'] - 1 or step % (cfg['num_steps'] // 2) == 0

    if cfg['policy_type'] == 'conditional' and not use_gt_cues:
      ppo.generator.create_context() # refresh generator context

    # release memory from experience collection
    if cfg["should_free_memory"]:
      free_memory([memory_loader, states, env, actions, log_probs, rewards, advantages, values, cues, dones])

    # 
    # periodic routines
    # 

    if (step + 1) % cfg["save_interval"] == 0:
      ppo.save(join(cfg["out_dir"], f"net_{step}.pth"))

    if step % cfg["test_interval"] == 0:
      print(f"""\n\n
        Step: {step}
        Rollout count: {rollout_count / cfg["epochs_per_step"]}
        Avg loss: {avg_loss / n_batches / curr_batch_size / cfg["epochs_per_step"]}
        Avg reward: {avg_rwd / n_batches / curr_batch_size / cfg["epochs_per_step"]}
        Avg action: {avg_log_probs / n_batches / curr_batch_size / cfg["epochs_per_step"]}""")

      #
      # rollout policy in test env
      #

      ppo.net.eval()
      if cfg['policy_type'] == 'conditional' and not use_gt_cues:
        ppo.generator.eval()

      with torch.no_grad():
        rew = 0
        accuracy = 0.0
        done = False
        count = 1
        self_state = ppo._to_tensor(test_env.reset(seed=cfg['seed'] + step))

        cues = ppo._to_tensor(test_env.compute_cues())

        frames = list()
        predicted = list()
        actual = list()
        #ys = list() # uncomment for debugging policy/generator net interaction
        #logs = list()

        while not done and count < EVAL_TIMEOUT:
          gt_cues = cues[:, icue[0]:icue[1]]

          if cfg['policy_type'] == 'conditional' and use_gt_cues:
            input_cues = gt_cues if cfg['z_noise_level'] == 0 else ppo._to_tensor(test_env.apply_gaussian_noise(test_env.compute_cues()[icue[0]:icue[1]], cfg['z_noise_level']))
            outputs = ppo.net(self_state, input_cues)
          elif cfg['policy_type'] == 'conditional' and not use_gt_cues:
            input_cues, _ = ppo.generator(self_state)
            outputs = ppo.net(self_state, input_cues.detach())
          else:
            input_cues = gt_cues
            outputs = ppo.net(self_state)

          policy = ppo.net.pi()
          env_action = policy.sample().detach()
          # NOTE: actions from non-categorical will keep batch dim;otherwise, will be 1D
          if ppo.net.action_type != 'discrete': env_action = env_action.squeeze(0)

          next_state, reward, done, info = test_env.step(env_action.cpu().numpy())

          val = outputs[0].detach().cpu().numpy().squeeze(0)
          gt_cues = gt_cues.detach().cpu().numpy()
          prediction = input_cues.detach().cpu().numpy() # for default policy or gt source, this is gt cue
          accuracy += 1 - MAE(prediction, gt_cues)
          rew += reward

          #logs.append(outputs[1].detach().cpu().numpy().squeeze(0))
          #ys.append(outputs[2].detach().cpu().numpy().squeeze(0))
          actual.append(gt_cues.squeeze(0))
          predicted.append(prediction.squeeze(0))
          if should_record_visual_output:
            cues = cues.detach().cpu().numpy().squeeze(0)
            cues[icue[0]:icue[1]] = prediction.squeeze(0)
            frames.append((count, test_env.draw_cues(z_type, cues)))

          count += 1
          self_state = ppo._to_tensor(next_state)
          cues = ppo._to_tensor(test_env.compute_cues())

      # save best model (overwrites previous trials)
      if rew > max_rew:
        ppo.save(join(cfg["out_dir"], f"net_best.pth"))
        pickle.dump([[step], [rew], [val]], open(join(cfg["out_dir"], f"net_best-t-{trial_id}.pkl"), 'wb'))
        max_rew = rew

      print(f"test reward: {rew}")
      print(f"value prediction: {val}")
      logger.log("rewards", rew, is_train=False)
      logger.log("accuracy", accuracy / count, is_train=False)
      ppo.save(join(cfg["out_dir"], f"net_final.pth"))
      for t in range(len(actual)):
        logger.log('predicted', predicted[t], is_train=False) # need 2d for logging
        logger.log('actual', actual[t], is_train=False)       # need 2d for logging
        #logger.log('ys', ys[t], is_train=False)               # need 2d for logging
        #logger.log('logs', logs[t], is_train=False)            # need 2d for logging

      logger.dump(cfg["out_dir"], trial_id)

      if should_record_visual_output:
        video.save_as_gif(frames, join(cfg["out_dir"], f"traj_{step}.gif"))

      # prepare models for training
      ppo.net.train()
      if cfg['policy_type'] == 'conditional' and not use_gt_cues:
        if ppo.generator.should_finetune: ppo.generator.train()
        ppo.generator.create_context() # refresh generator context

  # save final model (overwrite previous trials)
  # save training metrics
  logger.dump(cfg["out_dir"], trial_id)
  logger.reset()

  if cfg["should_free_memory"]:
    free_memory([memory_loader, ppo.net, ppo, test_env, env])

  return max_rew
