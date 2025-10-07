from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
import os
from .utils import set_seed

def train_ppo(
    make_env_fn,
    total_timesteps: int = 200_000,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    log_dir: str = "./logs",
    eval_env_fn=None,
    eval_freq: int = 10_000,
    seed: int | None = None,
    vec_norm_obs: bool = False,
    vec_norm_reward: bool = True,
    vec_clip_obs: float = 10.0,
    vec_clip_reward: float = 10.0,
    device: str = "cpu",
    sub_procs: int = 1,
):
    os.makedirs(log_dir, exist_ok=True)
    if seed is not None:
        set_seed(seed)

    if(sub_procs == 1): env = DummyVecEnv([make_env_fn(1)])
    else: env = SubprocVecEnv([make_env_fn(i) for i in range(sub_procs)])
    if vec_norm_obs or vec_norm_reward:
        env = VecNormalize(
            env,
            norm_obs=vec_norm_obs,
            norm_reward=vec_norm_reward,
            clip_obs=vec_clip_obs,
            clip_reward=vec_clip_reward,
        )
    model = PPO("MlpPolicy", env,
                learning_rate=learning_rate, gamma=gamma, gae_lambda=gae_lambda,
                clip_range=clip_range, ent_coef=ent_coef, vf_coef=vf_coef,
                n_steps=n_steps, batch_size=batch_size, n_epochs=n_epochs,
                verbose=1, tensorboard_log=log_dir, device=device)

    callback = None
    if eval_env_fn is not None:
        eval_env = DummyVecEnv([eval_env_fn])
        if isinstance(env, VecNormalize):
            # Wrap eval env with separate VecNormalize (uses running stats, but set to eval mode after training)
            eval_env = VecNormalize(
                eval_env,
                norm_obs=vec_norm_obs,
                norm_reward=False,  # don't normalize reward during eval callbacks
                clip_obs=vec_clip_obs,
                training=False,
            )
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, min_evals=5, verbose=1)
        callback = EvalCallback(eval_env, best_model_save_path=log_dir, log_path=log_dir,
                                eval_freq=eval_freq, deterministic=True, render=False, callback_after_eval=stop_cb)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    # If using VecNormalize, persist statistics for later reuse
    if isinstance(env, VecNormalize):
        env.save(os.path.join("logs/saves", "vecnormalize.pkl"))
    return model
