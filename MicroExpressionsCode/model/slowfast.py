import argparse
import itertools
import os
import sys
import time
import subprocess

sys.path.insert(0, '/home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/')
from main_dir.consts import SLOWFAST_RUN_NET_PATH
from model.utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9998",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="/home/khen_proj_1/PycharmProjects/slowfast_test/slowfast/configs/MicroExpressions/SLOWFAST_8x8_R50_stepwise_multigrid.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default="SLOWFAST.ALPHA 4 SLOWFAST.BETA_INV 8 TRAIN.BATCH_SIZE 16 SOLVER.BASE_LR 0.001 SOLVER.OPTIMIZING_METHOD adam OUTPUT_DIR /home/khen_proj_1/yuvaltuval/MicroExpressionsFaceRecognition/output/SlowFast/2022-08-20_21:00_SAMM_Grad_Cam",
        nargs=argparse.REMAINDER,
    )

    # parse arguments
    args = parser.parse_args()

    args_dict = dict()
    length = len(args.opts)
    i = 0
    while i < length:
        key = args.opts[i]
        values = []

        # all elements that are not all in upper case are values. if it's upper case than it's a new arg
        j = i + 1
        while j < length and not args.opts[j].isupper():
            values.append(args.opts[j])
            j += 1
        args_dict[key] = values
        i = j

    # getting permutations of all parameters
    keys, values = zip(*args_dict.items())
    permutations_dicts = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # add directory name based on the parameters
    for permutation in permutations_dicts:
        path = permutation['OUTPUT_DIR']
        temp_perm = permutation
        del temp_perm[key]
        folder_name = str(temp_perm).replace(" ", "_").replace(":", "").replace("{", "").replace("}", "").replace("'","").replace(",", "_").replace(".", "-")
        new_path = os.path.join(path, folder_name)
        permutation['OUTPUT_DIR'] = new_path

    # convert to list of lists
    permutation_lists = []
    for permutation in permutations_dicts:
        permutation_list = []
        for key, value in permutation.items():
            permutation_list.append(key)
            permutation_list.append(value)
        permutation_lists.append(permutation_list)

    # add SlowFast regular arguments and path to run_net.py
    start_of_list = ['python', SLOWFAST_RUN_NET_PATH, '--shard_id', str(args.shard_id), '--num_shards',
                     str(args.num_shards), '--init_method', str(args.init_method), '--cfg', args.cfg_file]
    permutation_lists = [start_of_list + permutation for permutation in permutation_lists]

    # unparse back to string
    commands = [" ".join(permutation) for permutation in permutation_lists]

    print(f"\n[INFO] received {len(commands)} different permutations. Starting...\n")
    Failed = 0
    for command in commands:
        print(f"[INFO] Running:\n{command}\n")
        # result = os.system(command)
        result = subprocess.call(command, shell=True)

        if 0 == result:
            print("Command Successfully")
        else:
            print("Command Failed")
            Failed += 1
        print("going to run next iteration in 10 seconds")
        time.sleep(10)
    if 0 == result:
        print(f"[INFO] finished running all commands with {Failed} Failures")
