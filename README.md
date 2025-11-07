# S3
S3 is a PyTorch implementation for the paper "S3: Stable Subgoal Selection by Constraining Uncertainty of Coarse Dynamics in Hierarchical Reinforcement Learning" 

<!-- ## Dependencies
- Python 3.6
- PyTorch 1.3
- OpenAI Gym
- MuJoCo
- Gymnasium (for Mujoco Reacher/Pusher tasks)

~~Also, to run the MuJoCo experiments, a license is required (see [here](https://www.roboti.us/license.html)).~~

## Usage

**Update:** implementation for discrete control tasks is in the `discrete/` folder; please refer to the usage therein.

### Training
- Ant Gather
```
python main.py --env_name AntGather
```
- Ant Maze
```
python main.py --env_name AntMaze
```
- Ant Maze Sparse
```
python main.py --env_name AntMazeSparse
```
- Reacher (Gymnasium Mujoco)
```
python main.py --env_name Reacher-v5
```
- Pusher (Gymnasium Mujoco)
```
python main.py --env_name Pusher-v5
```
### Evaluation
- Ant Gather
```
python eval.py --env_name AntGather --model_dir [MODEL_DIR]
```
- Ant Maze
```
python eval.py --env_name AntMaze --model_dir [MODEL_DIR]
```
- Ant Maze Sparse
```
python eval.py --env_name AntMazeSparse --model_dir [MODEL_DIR]
```
- Reacher (Gymnasium Mujoco)
```
python eval.py --env_name Reacher-v5 --model_dir [MODEL_DIR]
```
- Pusher (Gymnasium Mujoco)
```
python eval.py --env_name Pusher-v5 --model_dir [MODEL_DIR]
```
Default `model_dir` is `pretrained_models/`.

## Pre-trained models

See `pretrained_models/` for pre-trained models on all tasks. The expected performances of the pre-trained models are as follows (averaged over 100 evaluation episodes):

|Ant Gather|Ant Maze|Ant Maze Sparse|
|:--:|:--:|:--:|
|3.0|96%|89%|

## Citation
If you find this work useful in your research, please cite:
```
@inproceedings{zhang2020generating,
  title={Generating adjacency-constrained subgoals in hierarchical reinforcement learning},
  author={Zhang, Tianren and Guo, Shangqi and Tan, Tian and Hu, Xiaolin and Chen, Feng},
  booktitle={NeurIPS},
  year={2020}
}
``` -->
