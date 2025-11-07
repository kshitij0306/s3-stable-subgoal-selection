"""Entry point for training the S3 agent.

Inspired by https://github.com/trzhang0116/HRAC.
"""

import argparse

from s3.train import run_s3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default="s3", type=str)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--eval_freq", default=5e3, type=float)
    parser.add_argument("--max_timesteps", default=5e6, type=float)
    parser.add_argument("--model_dir", default="./models", type=str,
                        help="Directory to save model checkpoints.")
    parser.add_argument("--save_models", action="store_true")
    parser.add_argument("--save_halfway_checkpoint", action="store_true",
                        help="Also snapshot models once halfway through training.")
    parser.add_argument("--half_checkpoint_tag", default="half", type=str,
                        help="Suffix appended to halfway checkpoint files.")
    parser.add_argument("--final_checkpoint_tag", default="final", type=str,
                        help="Suffix appended to final checkpoint files when halfway saving is enabled.")
    parser.add_argument("--env_name", default="AntMaze", type=str)
    parser.add_argument("--load", default=False, type=bool)
    parser.add_argument("--log_dir", default="./logs", type=str)
    parser.add_argument("--no_correction", action="store_true")
    parser.add_argument("--inner_dones", action="store_true")
    parser.add_argument("--absolute_goal", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")
    parser.add_argument("--load_adj_net", default=False, action="store_true")
    parser.add_argument("--disable_adj_net", action="store_true",
                        help="Disable adjacency-network regularisation")

    parser.add_argument("--fast_mode", action="store_true",
                        help="Skip optional logging/adjacency extras and favour speed.")

    parser.add_argument("--gid", default=0, type=int)
    parser.add_argument("--traj_buffer_size", default=50000, type=int)
    parser.add_argument("--lr_r", default=2e-4, type=float)
    parser.add_argument("--r_margin_pos", default=1.0, type=float)
    parser.add_argument("--r_margin_neg", default=1.2, type=float)
    parser.add_argument("--r_training_epochs", default=25, type=int)
    parser.add_argument("--r_batch_size", default=64, type=int)
    parser.add_argument("--r_hidden_dim", default=128, type=int)
    parser.add_argument("--r_embedding_dim", default=32, type=int)
    parser.add_argument("--goal_loss_coeff", default=20., type=float)

    parser.add_argument("--manager_propose_freq", default=10, type=int)
    parser.add_argument("--train_manager_freq", default=10, type=int)
    parser.add_argument("--man_discount", default=0.99, type=float)
    parser.add_argument("--ctrl_discount", default=0.95, type=float)

    parser.add_argument("--man_soft_sync_rate", default=0.005, type=float)
    parser.add_argument("--man_batch_size", default=256, type=int)
    parser.add_argument("--man_buffer_size", default=2e5, type=int)
    parser.add_argument("--man_rew_scale", default=1.0, type=float)
    parser.add_argument("--man_act_lr", default=1e-4, type=float)
    parser.add_argument("--man_crit_lr", default=1e-3, type=float)
    parser.add_argument("--candidate_goals", default=10, type=int)

    parser.add_argument("--man_ctrl_rew_balance_start", default=0.0, type=float)
    parser.add_argument("--man_ctrl_rew_balance_end", default=0.5, type=float)
    parser.add_argument("--man_ctrl_rew_balance_steps", default=400_000, type=float)
    parser.add_argument("--reach_buffer_size", default=100_000, type=int)
    parser.add_argument("--reach_warmup_samples", default=8_000, type=int,
                        help="Minimum reach-buffer entries before enabling manager shaping")
    parser.add_argument("--reach_warmup_rounds", default=2, type=int,
                        help="Number of reach-net optimisation rounds before enabling shaping")

    parser.add_argument("--popart", action="store_true", help="Use Pop-Art normalisation in the controller critic")
    parser.add_argument("--ctrl_soft_sync_rate", default=0.005, type=float)
    parser.add_argument("--ctrl_batch_size", default=256, type=int)
    parser.add_argument("--ctrl_buffer_size", default=2e5, type=int)
    parser.add_argument("--ctrl_rew_scale", default=1.0, type=float)
    parser.add_argument("--ctrl_act_lr", default=1e-4, type=float)
    parser.add_argument("--ctrl_crit_lr", default=1e-3, type=float)

    parser.add_argument("--noise_type", default="normal", type=str)
    parser.add_argument("--ctrl_noise_sigma", default=1., type=float)
    parser.add_argument("--man_noise_sigma", default=0.6, type=float)
    parser.add_argument("--ctrl_noise_sigma_final", default=None, type=float,
                        help="Optional target sigma for controller noise annealing")
    parser.add_argument("--man_noise_sigma_final", default=None, type=float,
                        help="Optional target sigma for manager noise annealing")
    parser.add_argument("--noise_anneal_start", default=0, type=float,
                        help="Timestep to begin noise annealing")
    parser.add_argument("--noise_anneal_steps", default=0, type=float,
                        help="Duration (timesteps) over which to anneal noise")

    parser.add_argument("--n_mix", default=5, type=int, help="Number of mixtures in the MDN")

    parser.add_argument("--log_dispersion_stats", action="store_true",
                        help="Log full dispersion diagnostics (extra compute)")

    parser.add_argument("--enable_transformer_logging", action="store_true",
                        help="Enable logging to be stored in JSON results folder")

    # Dispersion control (which TSD metric to optimize)
    parser.add_argument("--dispersion",
                        choices=["var", "logdet", "maxeig", "anisotropy", "dir_perp", "w2", "chance", "cvar"],
                        default="var",
                        help="Dispersion metric for Manager PBRS: "
                            "var=trace(Σ), logdet=log|Σ|, maxeig=λ_max, anisotropy=λ_max/λ_min, "
                            "dir_perp=trace(Σ)-v^TΣv, w2=E||X-g||^2, chance=1-P(||X-g||<=eps), cvar=CVaRα(||X-g||).")

    parser.add_argument("--disp-eps", default=2.0, type=float,
                        help="ε for chance constraint: P(||X-g||<=ε)")
    parser.add_argument("--disp-alpha", default=0.1, type=float,
                        help="alpha for CVaR (tail probability)")
    parser.add_argument("--disp-samples", default=64, type=int,
                        help="Samples from MDN for chance/CVaR (per decision)")

    parser.add_argument("--freeze_worker", action="store_true",
                        help="Freeze controller (worker) policy and load weights from a pretrained checkpoint.")
    parser.add_argument("--worker_model_dir", default="./models", type=str,
                        help="Directory containing pretrained worker checkpoints.")
    parser.add_argument("--worker_algo", default=None, type=str,
                        help="Algorithm label used when saving the pretrained worker (defaults to current algo).")
    parser.add_argument("--worker_env_name", default=None, type=str,
                        help="Environment name used to train the pretrained worker (defaults to current env).")
    parser.add_argument("--worker_checkpoint_tag", default=None, type=str,
                        help="Optional suffix for selecting a specific worker checkpoint (e.g. 'half').")
    parser.add_argument("--save_periodic", action="store_true",
                        help="Enable periodic checkpoint archiving every 1M steps.")

    args = parser.parse_args()

    if args.fast_mode:
        args.disable_adj_net = True
        args.log_dispersion_stats = False

    args.enable_transformer_logging = False

    if args.env_name == "AntGather":
        args.inner_dones = True

    env_base_scale = {
        "AntMaze": 0.1,
        "AntMazeSparse": 1.0,
        "AntPush": 0.15,
        "AntFall": 0.15,
        "AntGather": 1.0,
        "Reacher-v5": 0.1,
        "Pusher-v5": 0.1,
    }

    args.man_rew_scale = max(0.0, min(1.0, args.man_rew_scale))
    args.man_env_scale = env_base_scale.get(args.env_name, 0.1)
    args.use_adj_net = not args.disable_adj_net

    print('=' * 30)
    for key, val in vars(args).items():
        print('{}: {}'.format(key, val))

    run_s3(args)
