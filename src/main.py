#########################################################################
# Sign in to Hugging Face by running this command: huggingface-cli login
# 
# Set project location to be able to call project modules 
import argparse
import sys
sys.path.append("/mnt/isgnas/home/anl31/documents/code/ai-vul-detect-pairwise-learning-to-rank")
#########################################################################

from load_data_pairs import load_pairs_from_jsonl


# TODO: Remove default values:
primevul_paired_train_data_file = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_train_paired.jsonl"
primevul_paired_test_data_file = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_test_paired.jsonl"
primevul_paired_valid_data_file = "/mnt/isgnas/home/anl31/documents/data/PrimeVul_v0.1/primevul_valid_paired.jsonl"


def main(args):
    """
    Load data, train the model, evaluate results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """

    # Load the dataset
    train_data = load_pairs_from_jsonl(args.primevul_paired_train_data_file)
    
    print(f"Extracted {len(train_data)} pairs of vulnerable and benign code.")
    # Example of accessing the first pair
    if train_data:
        print("Example Pair:")
        print("Vulnerable Code:")
        print(train_data[0][0])
        print("Benign Code:")
        print(train_data[0][1])



# Example usage
if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Pairwise Learning to Rank Vulnerability Detection")
    
    parser.add_argument('-c', '--compute', default="cuda", 
                        help='Type of compute to use (cpu, cuda, mps...)')
    
    # Commands related to loading the dataset
    parser.add_argument("--primevul_paired_train_data_file", default=primevul_paired_train_data_file, type=str, 
                        help="Path to the PrimeVul paired train data file.")
    parser.add_argument("--primevul_paired_train_test_file", default=primevul_paired_test_data_file, type=str, 
                        help="Path to the PrimeVul paired test data file.")
    parser.add_argument("--primevul_paired_valid_data_file", default=primevul_paired_valid_data_file, type=str, 
                        help="Path to the PrimeVul paired valid data file.")
    

    # Parse arguments and run main
    args = parser.parse_args()
    main(args)