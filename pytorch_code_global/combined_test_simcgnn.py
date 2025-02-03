from model import *
from utils import Data, split_validation
import torch
import random
import pickle
import argparse
import tqdm

import pandas as pd
import numpy as np
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="diginetica",
        help="dataset name: diginetica/yoochoose1_4/yoochoose1_64/sample",
    )
    parser.add_argument("--batchSize", type=int, default=512, help="input batch size")
    parser.add_argument("--hiddenSize", type=int, default=100, help="hidden state size")
    parser.add_argument(
        "--epoch", type=int, default=30, help="the number of epochs to train for"
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate"
    )  # [0.001, 0.0005, 0.0001]
    parser.add_argument(
        "--lr_dc", type=float, default=0.1, help="learning rate decay rate"
    )
    parser.add_argument(
        "--lr_dc_step",
        type=int,
        default=10,
        help="the number of steps after which the learning rate decay",
    )  # 原本是3、10
    parser.add_argument(
        "--l2", type=float, default=1e-5, help="l2 penalty"
    )  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
    parser.add_argument("--step", type=int, default=1, help="gnn propogation steps")
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="the number of epoch to wait before early stop ",
    )
    parser.add_argument(
        "--nonhybrid",
        action="store_true",
        help="only use the global preference to predict",
    )
    parser.add_argument("--validation", action="store_true", help="validation")
    parser.add_argument("--seed", type=int, default=3407, help="random seed")
    parser.add_argument(
        "--n_layers", type=int, default=2, help="num of layers of lintransformer"
    )
    parser.add_argument(
        "--valid_portion",
        type=float,
        default=0.1,
        help="split the portion of training set as validation set",
    )
    parser.add_argument(
        "--contrast_loss_weight",
        type=float,
        default=0.2,
        help="Weight of contrastive loss",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Initial temperature for InfoNCE loss",
    )
    parser.add_argument(
        "--temperature_decay",
        type=float,
        default=0.9,
        help="Decay rate for temperature",
    )
    parser.add_argument(
        "--min_temperature", type=float, default=0.01, help="Minimum temperature value"
    )
    parser.add_argument(
        "--num_neg", type=int, default=8, help="Number of negative samples per session"
    )
    parser.add_argument(
        "--similarity_threshold", type=float, default=0.1, help="Similarity threshold"
    )
    parser.add_argument(
        "--fusion_factor", type=float, default=0.8, help="fusion factor"
    )
    parser.add_argument(
        "--item_id", type=int, default=33889, help="Item ID to analyze in test data"
    )
    parser.add_argument(
        "--num_runs", type=int, default=5, help="Number of prediction runs to average"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./output/diginetica/epoch_7.pth",
        help="Path to the trained model",
    )
    opt = parser.parse_args()
    opt.item_id = 33889
    opt.num_runs = 5
    opt.model_path = "./output/diginetica/epoch_21.pth"
    print(opt)

    # Set random seeds for reproducibility
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    # Load and preprocess training data
    train_data_raw = pickle.load(open(f"../datasets/{opt.dataset}/train.txt", "rb"))
    train_data = Data(
        train_data_raw,
        shuffle=True,
        num_neg=opt.num_neg,
        similarity_threshold=opt.similarity_threshold,
    )

    # Load and preprocess test data
    test_data_raw = pickle.load(open(f"../datasets/{opt.dataset}/test.txt", "rb"))
    test_data = Data(
        test_data_raw,
        shuffle=False,
        num_neg=opt.num_neg,
        similarity_threshold=opt.similarity_threshold,
    )

    # Determine the maximum sequence length
    opt.len_max = max(train_data.len_max, test_data.len_max)
    # Set the number of nodes (items)
    n_node = 43098  # Adjust this based on your dataset

    # Initialize the model and load pre-trained weights
    model = trans_to_cuda(SessionGraph(opt, n_node))
    model.load_state_dict(torch.load(opt.model_path))
    print("---model loaded---")

    model.eval()

    # Initialize a list to store cumulative scores
    cumulative_scores = None

    # Number of prediction runs
    num_runs = opt.num_runs
    print(f"Running {num_runs} prediction runs and averaging the results.")

    # Run predictions multiple times and accumulate scores
    for run in range(num_runs):
        print(f"Starting prediction run {run + 1}/{num_runs}")
        slices = test_data.generate_batch(model.batch_size)
        run_scores = []

        for i in tqdm.tqdm(slices, desc=f"Run {run + 1}"):
            targets, scores, _ = forward(model, i, test_data)
            run_scores.append(scores.detach().cpu().numpy())

        run_scores = np.vstack(run_scores)  # Shape: (num_samples, n_nodes)

        if cumulative_scores is None:
            cumulative_scores = run_scores
        else:
            cumulative_scores += run_scores

    # Calculate the average scores
    average_scores = cumulative_scores / num_runs

    # Save the averaged scores if needed
    np.save(f"average_scores.{opt.dataset}.npy", average_scores)
    print(f"Averaged scores saved to average_scores.{opt.dataset}.npy")

    # Analyze and compare with true distribution for a specific item_id
    item_id_to_analyze = opt.item_id
    print(f"Analyzing sessions ending with item ID: {item_id_to_analyze}")

    # Extract true targets and predicted top-1 items for sessions ending with the specified item_id
    res_true = []
    res_pred = []
    last_items = np.array(test_data.last_items)

    # Identify indices where the last item is the specified item_id
    target_indices = np.where(last_items == item_id_to_analyze)[0]

    # Extract the true targets
    true_targets = (
        np.array(test_data.targets)[target_indices] - 1
    )  # Adjusting if targets were decremented

    # Extract the predicted top-1 items from the averaged scores
    predicted_top1 = average_scores[target_indices].argmax(axis=1)

    # Adjusting back to original item IDs if needed
    predicted_top1 += 1  # Assuming item IDs start from 1

    # Collect the results
    res_true = true_targets.tolist()
    res_pred = predicted_top1.tolist()

    # Count the frequency of each target and prediction
    from collections import Counter

    res_cnt_true = Counter(res_true)
    res_cnt_pred = Counter(res_pred)

    print(
        "True Target Counts (for sessions ending with item ID {}):".format(
            item_id_to_analyze
        )
    )
    print(res_cnt_true)
    print("Total True Targets:", sum(res_cnt_true.values()))
    print(
        "\nPredicted Counts (Top-1, for sessions ending with item ID {}):".format(
            item_id_to_analyze
        )
    )
    print(res_cnt_pred)
    print("Total Predictions:", sum(res_cnt_pred.values()))

    # Save the counts for further analysis if needed
    np.save("gt_cnt.npy", res_cnt_true)
    np.save("simcgnn_cnt.npy", res_cnt_pred)
    print("True target counts saved to gt_cnt.npy")
    print("Predicted counts saved to simcgnn_cnt.npy")

    # Optionally, you can compute additional metrics or visualize the distribution
    # For example, print top 10 most common predictions
    print("\nTop 10 True Targets:")
    for item, count in res_cnt_true.most_common(10):
        print(f"Item ID {item}: {count} times")

    print("\nTop 10 Predicted Items:")
    for item, count in res_cnt_pred.most_common(10):
        print(f"Item ID {item}: {count} times")


if __name__ == "__main__":
    main()
