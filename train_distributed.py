import argparse
import os
import subprocess
import sys
import time


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _get_cur_time_str(formatter: str = "%m-%d_%H:%M:%S") -> str:
    time_stamp = time.time()
    time_array = time.localtime(time_stamp)
    return time.strftime(formatter, time_array)


def parse_args():
    """
    Helper function parsing the command line options
    @retval ArgumentParser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rapid_pL1')
    parser.add_argument('--backbone', type=str, default='dark53')
    parser.add_argument('--dataset', type=str, default='COCO')
    parser.add_argument('--batch_size', type=int, default=1)

    parser.add_argument('--high_resolution', action='store_true')

    parser.add_argument('--checkpoint', type=str, default='')

    parser.add_argument('--eval_interval', type=int, default=1000)
    parser.add_argument('--img_interval', type=int, default=500)
    parser.add_argument('--print_interval', type=int, default=1)
    parser.add_argument('--checkpoint_interval', type=int, default=2000)

    parser.add_argument('--debug', action='store_true')

    parser.add_argument("--gpu_id",
                        type=str,
                        default="0,1,2,3,4,5,6,7",
                        help="gpu id for train")
    return parser.parse_args()


def main():
    args = parse_args()

    # world size in terms of number of processes
    num_gpus = len(args.gpu_id.split(","))
    dist_world_size = num_gpus

    # set PyTorch distributed related environmental variables
    current_env = os.environ.copy()
    current_env["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    current_env["MASTER_ADDR"] = "127.0.0.1"
    current_env["MASTER_PORT"] = str(_find_free_port())
    current_env["WORLD_SIZE"] = str(dist_world_size)
    current_env["CURRENT_TIME"] = _get_cur_time_str()  # for logger

    processes = []

    if "MP_NUM_THREADS" not in os.environ and num_gpus > 1:
        current_env["OMP_NUM_THREADS"] = str(1)
        # print("*****************************************\n"
        #       "Setting OMP_NUM_THREADS environment variable for each process "
        #       "to be {} in default, to avoid your system being overloaded, "
        #       "please further tune the variable for optimal performance in "
        #       "your application as needed. \n"
        #       "*****************************************".format(current_env["OMP_NUM_THREADS"]))

    training_script = "train.py"
    for local_rank in range(num_gpus):
        # each process's rank
        dist_rank = local_rank
        current_env["RANK"] = str(dist_rank)
        current_env["LOCAL_RANK"] = str(local_rank)

        # spawn the processes
        cmd = [
            sys.executable,
            "-u", training_script,
            f"--model={args.model}",
            f"--backbone={args.backbone}",
            f"--dataset={args.dataset}",
            f"--batch_size={args.batch_size}",
            f"--checkpoint={args.checkpoint}",
            f"--eval_interval={args.eval_interval}",
            f"--img_interval={args.img_interval}",
            f"--print_interval={args.print_interval}",
            f"--checkpoint_interval={args.checkpoint_interval}",
            f"--local_rank={local_rank}",
            f"--multi_gpu",
        ]
        if args.high_resolution:
            cmd.append("--high_resolution")
        if args.debug:
            cmd.append("--debug")

        process = subprocess.Popen(cmd, env=current_env)
        processes.append(process)

    for process in processes:
        process.wait()
        if process.returncode != 0:
            raise subprocess.CalledProcessError(returncode=process.returncode, cmd=process.args)


if __name__ == "__main__":
    main()
