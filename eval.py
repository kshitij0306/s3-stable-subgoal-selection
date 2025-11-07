import argparse

from s3.eval import eval_s3


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--algo", default="s3", type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--gid", type=int, default=0)
    parser.add_argument("--env_name", default="AntMaze", type=str)
    parser.add_argument("--eval_episodes", default=100, type=int)
    parser.add_argument("--load", default=True, type=bool)
    parser.add_argument("--model_dir", default="./pretrained_models", type=str)
    parser.add_argument("--checkpoint", default="base", type=str,
                        help="Checkpoint suffix to load; use 'base' for latest weights.")
    parser.add_argument("--manager_propose_freq", default=10, type=int)
    parser.add_argument("--absolute_goal", action="store_true")
    parser.add_argument("--binary_int_reward", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video_dir", default="./videos", type=str)
    parser.add_argument("--sample_n", default=0, type=int, help="If >0, sample N landings s_{t+c} and save a plot instead of eval")
    parser.add_argument("--plots_dir", default="./plots", type=str, help="Output directory for saved scatter plots")
    parser.add_argument("--st", nargs="+", type=float, default=None, help="Full state vector for s_t (space separated floats)")
    parser.add_argument("--gt", nargs="+", type=float, default=None, help="Goal vector g_t (space separated floats)")
    parser.add_argument("--bg_alpha", type=float, default=0.60, help="Alpha for maze background overlay (0..1)")

    parser.add_argument("--heatmap", action="store_true")
    args = parser.parse_args()

    eval_s3(args)
