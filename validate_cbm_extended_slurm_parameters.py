import sys

from utils.argparser import (
    cbm_extended_argument_config,
    cbm_extended_argument_parser,
    cbm_extended_argument_validator,
)

if len(sys.argv) < 3:
    print("Invoke with python 'python validate_cbm_extended_slurm_parameters.py <start_id> <end_id>'")
    exit()

start_id = int (sys.argv[1])
end_id = int(sys.argv[2])

print(f"Checking SLURM-Jobs from {start_id} to {end_id}")

for slurm_id in range(start_id, end_id+1):
    try:
        parser = cbm_extended_argument_parser()
        cmd = cbm_extended_argument_config(slurm_id)[3:]
        args = parser.parse_args(cmd)

        cbm_extended_argument_validator(args)
    
    except SystemExit as e:
        # argparse error
        print(f"[ARGPARSE ERROR] Slurm ID {slurm_id}: exit code {e.code}")

    except Exception as e:
        # Validator errors etc.
        print(f"[ERROR] Slurm ID {slurm_id}: {e}")

    else:
        print(f"ID {slurm_id} positive")