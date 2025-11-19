from gym.envs.registration import register

# custom car racing  (similar to CartPole-v1 env.unwrapped.spec())
register(
    id='CarRacing-v1r',
    entry_point='env.custom:CarRacingMod',
    max_episode_steps=1000,
    reward_threshold=900,
    nondeterministic=False,
    autoreset=False
)

# custom cart pole (similar to CartPole-v1 env.unwrapped.spec())
register(
    id='CartPole-v1r',
    entry_point='env.custom:CartPoleMod',
    max_episode_steps=500,
    reward_threshold=475,
    nondeterministic=False,
    autoreset=False
)
