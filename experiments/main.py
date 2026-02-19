import torch
from datetime import datetime
import json
import os

from activation_function import main as run_activation_experiment

# Activation functions to test
ACTIVATION_FUNCTIONS = ["relu",
                        "leaky_relu",
                        "gelu",
                        "elu",
                        "selu",]

NUM_RUNS_PER_FUNCTION = 7
MAX_EPOCHS = 20
TARGET_ACCURACY = 0.80
DATA_PATH = "./data"


def run_experiments():
    """Run multiple activation function experiments with timestamp tracking."""

    results = {}

    for activation_fn in ACTIVATION_FUNCTIONS:
        print(f"\n{'='*60}")
        print(f"Testing {activation_fn.upper()} activation function")
        print(f"{'='*60}\n")

        runs_data = {
            "activation_function": activation_fn,
            "runs": [],
            "status": f"{0}/{NUM_RUNS_PER_FUNCTION} successful"
        }

        for run_num in range(1, NUM_RUNS_PER_FUNCTION + 1):
            print(
                f"\n>>> Run {run_num}/{NUM_RUNS_PER_FUNCTION} for {activation_fn}")

            # Track start time
            start_time = datetime.now()
            start_str = start_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

            try:
                # Set output directory for this run
                output_dir = os.path.join(
                    "experiments", activation_fn, f"run_{run_num}")

                # Run the experiment
                run_activation_experiment(
                    activation_fn_name=activation_fn,
                    output_dir=output_dir,
                    epochs=MAX_EPOCHS,
                    desired_accuracy=TARGET_ACCURACY,
                    capture_batches=1,
                    data_path=DATA_PATH
                )

                # Track end time
                end_time = datetime.now()
                end_str = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                duration = (end_time - start_time).total_seconds()

                # Record run data
                run_data = {
                    "run_number": run_num,
                    "start": start_str,
                    "end": end_str,
                    "duration_seconds": duration,
                    "status": "successful"
                }

                runs_data["runs"].append(run_data)

                print(f"✓ Run {run_num} completed in {duration:.2f}s")

            except Exception as e:
                end_time = datetime.now()
                end_str = end_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                duration = (end_time - start_time).total_seconds()

                run_data = {
                    "run_number": run_num,
                    "start": start_str,
                    "end": end_str,
                    "duration_seconds": duration,
                    "status": "failed",
                    "error": str(e)
                }

                runs_data["runs"].append(run_data)

                print(f"✗ Run {run_num} failed: {e}")

        # Update status
        successful_runs = sum(
            1 for run in runs_data["runs"] if run["status"] == "successful")
        runs_data["status"] = f"{successful_runs}/{NUM_RUNS_PER_FUNCTION} successful"

        # Calculate average time for successful runs
        successful_durations = [run["duration_seconds"]
                                for run in runs_data["runs"] if run["status"] == "successful"]
        if successful_durations:
            avg_time = sum(successful_durations) / len(successful_durations)
            runs_data["average_time_seconds"] = avg_time

        results[activation_fn] = runs_data

    return results


def print_results(results):
    """Pretty print the experiment results."""
    print(f"\n\n{'='*60}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*60}\n")

    for activation_fn, data in results.items():
        print(f"{activation_fn.upper()}:")
        print(f"  Status: {data['status']}")

        if "average_time_seconds" in data:
            print(
                f"  Average Time: {data['average_time_seconds']:.2f} seconds")

        # Calculate total time
        total_time = sum(run["duration_seconds"] for run in data["runs"])
        print(f"  Total Time: {total_time:.2f} seconds")

        print(f"  Run Timestamps:")
        for run in data["runs"]:
            print(f"    Run {run['run_number']}:")
            print(f"      Start: {run['start']}")
            print(f"      End:   {run['end']}")
            print(f"      Duration: {run['duration_seconds']:.2f}s")
            if run["status"] == "failed":
                print(f"      Error: {run['error']}")

        print()


def save_results(results, output_file="experiment_results.json"):
    """Save results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    print("Starting Activation Function Comparison Experiments")
    print(f"Testing: {', '.join(ACTIVATION_FUNCTIONS)}")
    print(f"Runs per function: {NUM_RUNS_PER_FUNCTION}")
    print(f"Max epochs: {MAX_EPOCHS}")
    print(f"Target accuracy: {TARGET_ACCURACY}")

    # Run experiments
    results = run_experiments()

    # Display results
    print_results(results)

    # Save results
    save_results(results)
