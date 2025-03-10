import pandas as pd
import numpy as np
import subprocess
import json
import os
from typing import List, Dict, Any, Tuple
import click
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO


def run_training(
    params: pd.DataFrame, idx: int, base_args: Dict[str, Any]
) -> np.ndarray:

    param_dict = params.iloc[0].to_dict()
    # Prepare command line arguments
    cmd_args = []
    for k, v in base_args.items():
        cmd_args.extend([f"--{k}", str(v)])

    # Add optimizer parameters
    for k, v in param_dict.items():
        cmd_args.extend([f"--{k}", str(v)])

    # Add run name with index
    run_name = f"{base_args.get('run_name', 'hebo_opt')}_{idx}"
    cmd_args.extend(["--run_name", run_name])

    print(f"Running iteration {idx} with parameters: {param_dict}")

    # Run the training process
    try:
        result = subprocess.run(
            ["torchrun", "--standalone", "--nproc-per-node=8", "train.py"] + cmd_args,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        with open("temp_output.json", "r") as f:
            checkpoint = json.load(f)

        assert checkpoint["run_name"] == run_name
        loss = checkpoint["val_loss"]

        print(f"Validation loss: {loss}")
    except subprocess.CalledProcessError as e:
        print(f"Training process failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        loss = 100.0  # Default high loss if process fails

    # Save the results for this iteration
    results = {"iteration": idx, "parameters": param_dict, "loss": float(loss)}

    with open(f"run_{idx}.json", "w") as f:
        json.dump(results, f, indent=2)

    return np.array([[loss]])


@click.command()
@click.option("--n_iterations", default=300, help="Number of optimization iterations")
@click.option(
    "--n_suggestions", default=1, help="Number of parameter suggestions per iteration"
)
@click.option(
    "--base_config", default="base_config.json", help="Base configuration file"
)
def optimize(n_iterations: int, n_suggestions: int, base_config: str):

    # Load base configuration
    if os.path.exists(base_config):
        with open(base_config, "r") as f:
            base_args = json.load(f)
    else:
        print(f"Base config file {base_config} not found, using defaults")
        base_args = {
            "project_name": "nanoMDM_hebo",
            "train_data": "/home/ubuntu/simo/0306/nano-llada/process_fineweb/fineweb_edu_shards/shard_*.bin",
            "val_data": "/home/ubuntu/simo/0306/nano-llada/process_fineweb/fineweb_edu_shards/val_shard_*.bin",
            "global_batch_size": 32 * 8,
            "per_gpu_batch_size": 32,
            "num_iterations": 1004,
            "warmup_iters": 100,
            "warmdown_iters": "20%",
            "val_every": 100,
            "save_every": 1000,
            "n_embed": 768,
            "vres": True,
            "n_layer": 12,
            "n_head": 6,
            "ff_expand": 4,
            "sequence_length": 1024,
            "tags": "",
            "do_compile": True,
        }

    # Define the parameter space in log scale for positive parameters
    RANGE_LB = 0.01
    RANGE_UB = 10
    space_params = [
        # Learning rate parameters
        {
            "name": "learning_rate",
            "type": "pow",
            "lb": 0.1 * RANGE_LB,
            "ub": 0.1 * RANGE_UB,
        },  # RANGE_LB to 10
        {
            "name": "weight_decay",
            "type": "pow",
            "lb": 0.1 * RANGE_LB,
            "ub": 0.1 * RANGE_UB,
        },  # RANGE_LB to 10
        # LR weight parameters
        {
            "name": "lr_wtexweight",
            "type": "pow",
            "lb": RANGE_LB,
            "ub": RANGE_UB,
        },  # RANGE_LB to 1
        {"name": "lr_attnxc_qxweight", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {"name": "lr_attnxc_kxweight", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {"name": "lr_attnxc_vxweight", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {
            "name": "lr_attnxc_projxweight",
            "type": "pow",
            "lb": RANGE_LB,
            "ub": RANGE_UB,
        },
        {"name": "lr_mlpxc_fcxweight", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {"name": "lr_mlpxc_projxweight", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {"name": "lr_lm_headxweight", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {"name": "lr_lamb1", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {"name": "lr_lamb2", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        # Init std parameters
        {"name": "initstd_wtexweight", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {
            "name": "initstd_attnxc_qxweight",
            "type": "pow",
            "lb": RANGE_LB,
            "ub": RANGE_UB,
        },
        {
            "name": "initstd_attnxc_kxweight",
            "type": "pow",
            "lb": RANGE_LB,
            "ub": RANGE_UB,
        },
        {
            "name": "initstd_attnxc_vxweight",
            "type": "pow",
            "lb": RANGE_LB,
            "ub": RANGE_UB,
        },
        {
            "name": "initstd_attnxc_projxweight",
            "type": "pow",
            "lb": 0.1 * RANGE_LB,
            "ub": 0.1 * RANGE_UB,
        },
        {
            "name": "initstd_mlpxc_fcxweight",
            "type": "pow",
            "lb": RANGE_LB,
            "ub": RANGE_UB,
        },
        {
            "name": "initstd_mlpxc_projxweight",
            "type": "pow",
            "lb": 0.1 * RANGE_LB,
            "ub": 0.1 * RANGE_UB,
        },
        {
            "name": "initstd_lm_headxweight",
            "type": "pow",
            "lb": RANGE_LB,
            "ub": RANGE_UB,
        },
        {"name": "initstd_lamb1", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
        {"name": "initstd_lamb2", "type": "pow", "lb": RANGE_LB, "ub": RANGE_UB},
    ]

    # Create design space
    space = DesignSpace().parse(space_params)

    # Initialize optimizer
    opt = HEBO(space)

    # Initialize best observation
    best_loss = float("inf")
    best_params = None
    # Run optimization
    for i in range(n_iterations):
        # Get suggestions
        suggestions = opt.suggest(n_suggestions=n_suggestions)

        # Evaluate suggestions
        losses = []
        for j in range(n_suggestions):
            suggestion_df = suggestions.iloc[[j]].reset_index(drop=True)
            loss = run_training(suggestion_df, i * n_suggestions + j, base_args)
            losses.append(loss)

            # Update best observation
            if loss[0][0] < best_loss:
                best_loss = loss[0][0]
                best_params = suggestion_df.iloc[0].to_dict()

                # Convert log space to linear space for reporting
                best_params_linear = {k: v for k, v in best_params.items()}

                print(f"New best loss: {best_loss}")
                print(f"New best parameters: {best_params_linear}")

                # Save best parameters
                with open("best_params.json", "w") as f:
                    json.dump(
                        {"loss": float(best_loss), "parameters": best_params_linear},
                        f,
                        indent=2,
                    )

        # Convert list of numpy arrays to a single numpy array
        losses = np.vstack(losses)

        # Observe losses
        opt.observe(suggestions, losses)

        print(f"Iteration {i+1}/{n_iterations} completed")
        print(f"Best loss so far: {best_loss}")


if __name__ == "__main__":
    optimize()
