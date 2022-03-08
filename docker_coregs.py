import os
import subprocess
import sys

CONTAINER = "lcford/coregs:1.0.1"


def get_args():
    try:
        return sys.argv[1:]
    except IndexError:
        return []


def get_cwd():
    return os.path.abspath(os.curdir)


def make_mount_calls(cwd):
    mount_dirs = [
        "graps_input",
        "graps_output",
        "objective_output",
        "generation_output",
        "data",
    ]
    calls = []
    for mdir in mount_dirs:
        calls.append("-v")
        calls.append(f"{cwd}/{mdir}:/code/coregs/{mdir}")
    return calls


def run_coregs_interactive(args, mount_calls):
    call = [
        "docker",
        "run",
        "-it",
        "--rm",
        *mount_calls,
        CONTAINER,
        "/bin/bash",
    ]
    print(" ".join(call))
    return subprocess.call(call)


def run_coregs(args, mount_calls):
    call = [
        "docker",
        "run",
        "--rm",
        *mount_calls,
        CONTAINER,
        "./docker_run_coregs.sh",
        *args,
    ]
    print(" ".join(call))
    return subprocess.call(call)


def main():
    args = get_args()
    cwd = get_cwd()
    mount_calls = make_mount_calls(cwd)
    exit_status = run_coregs(args, mount_calls)
    # exit_status = run_coregs_interactive(args, mount_calls)


if __name__ == "__main__":
    main()
