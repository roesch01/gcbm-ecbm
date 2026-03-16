import os
import subprocess

from utils.argparser import cbm_extended_argument_config

if __name__ == "__main__":

    # SLURM_ARRAY_TASK_ID is normally set by SLURM.
    # For local testing, you can manually set an ID here (e.g., 0).
    SLURM_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


    cmd = cbm_extended_argument_config(SLURM_ID)

    print("Running:")
    print(" ".join(cmd))

    result = subprocess.run(cmd, check=True)

    print(f"Done. Return Code: {result.returncode}")
