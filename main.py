import argparse
import os


iden_vln_path = os.getcwd()
project_path = os.path.dirname(iden_vln_path)


def ensure_output_dirs():
    if not os.path.exists(iden_vln_path + '/task'):    # high-level IDEN-VLN tasks
        os.makedirs(iden_vln_path + '/task')
    if not os.path.exists(iden_vln_path + '/step_task'):    # step-level IDEN-VLN tasks
        os.makedirs(iden_vln_path + '/step_task')
    if not os.path.exists(iden_vln_path + '/logs'):    # runtime logs
        os.makedirs(iden_vln_path + '/logs')

def read_args():
    parser = argparse.ArgumentParser(
        description="IDEN-VLN dataset construction pipeline with OpenAI-compatible LLM support."
    )
    parser.add_argument('--API_KEY', type=str, help="API key for the OpenAI-compatible LLM endpoint")
    parser.add_argument('--API_BASE', type=str,
        default=os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1'),
        help='OpenAI-compatible base URL, for example http://localhost:8000/v1 or http://localhost:11434/v1')

    parser.add_argument('--MODEL_MAIN', type=str,
        default=os.environ.get('IDENVLN_MODEL_MAIN', 'Qwen2.5-7B-Instruct'),
        help='Model name for forward task-level generation')

    parser.add_argument('--MODEL_STEP', type=str,
        default=os.environ.get('IDENVLN_MODEL_STEP', 'Qwen2.5-7B-Instruct'),
        help='Model name for backward or step-level instruction generation')

    parser.add_argument('--MODEL_VISION', type=str,
        default=os.environ.get('IDENVLN_MODEL_VISION', 'Qwen2.5-7B-Instruct'),
        help='Optional vision-capable model name; text-only Qwen endpoints can ignore this path')

    parser.add_argument('--TEMPERATURE', type=float, default=0.2,
        help='LLM sampling temperature; lower values are more deterministic')

    parser.add_argument('--MAX_TOKENS', type=int, default=1024,
        help='Maximum output tokens per LLM call')

    # generate task
    parser.add_argument('--scene_path', type=str, default=iden_vln_path + '/scene/', help="root scene path")
    parser.add_argument('--prompt_path', type=str, default=iden_vln_path + '/prompt/', help="root prompt path")
    parser.add_argument('--task_path', type=str, default=iden_vln_path + '/task/', help="root high-level task path")
    parser.add_argument('--region_file', type=str, default=iden_vln_path + '/scene/Per_Scene_Region_Weighted_Votes.csv', help="region semantic file")
    parser.add_argument('--loop', type=int, default=100, help="number of task configurations to generate")
    
    parser.add_argument('--scene_id', type=str, default=None, help="Whether or not to select a specific scene")
    parser.add_argument('--sample_region', type=bool, default=False, help="whether to sample rooms in a scene")
    parser.add_argument('--sample_obj', type=bool, default=True, help="whether to sample objects in a room")

    # simulation
    parser.add_argument('--scene', type=str, default=project_path + '/data/hm3d/', help="scene path")
    parser.add_argument('--scene_dataset', type=str, default=project_path + '/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json', help="scene dataset path")
    parser.add_argument('--max_step', type=int, default=500, help="maximum rollout steps to record")
    parser.add_argument('--success_dis', type=float, default=1, help="goal distance threshold considered successful")

    # step
    parser.add_argument('--step_task_path', type=str, default=iden_vln_path + '/step_task/', help="root step-task path")
    parser.add_argument('--ram_model', type=str, default=project_path + '/data/models/ram_plus_swin_large_14m.pth', help="path for ram model")
    parser.add_argument('--ram_logs', type=str, default=iden_vln_path + '/logs/step_task_logs.txt', help="path to save logs")
    parser.add_argument('--split_save_path', type=str, default=iden_vln_path + '/task/trail_list.txt', help="path to save the split trajectory")


    args = parser.parse_args()
    return args


def main():
    args = read_args()
    ensure_output_dirs()

    from task_gen import gen_task
    from dataset_gen import gen_traj
    from split_task import split_traj, gen_step_task

    # step 1: Generate task
    for i in range(args.loop):
        gen_task(args)

    # step 2: Generate trajectory
    gen_traj(args)

    # step 3: Split trajectory into segments
    split_traj(args)

    # step 4: Generate step-by-step task
    gen_step_task(args)

if __name__ == '__main__':
    main()
