from gym.envs.registration import register

register(
    id='reaching-v0',
    entry_point='gym_2d.envs:ReachingEnv',
)

register(
    id='reaching-v1',
    entry_point='gym_2d.envs:ReachingEnv_v1',
)

register(
    id='reaching-v2',
    entry_point='gym_2d.envs:ReachingEnv_v2',
)

#register(
#    id='mnist-v0',
#    entry_point='gym_cstr.envs:MnistEnv',
#)
