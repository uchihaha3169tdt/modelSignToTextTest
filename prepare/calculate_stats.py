import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def calculate_statistics(dataset_name="YOUTUBE_SIGN"):
    """
    Calculate mean and std statistics for normalization
    Input: .npy motion files with shape [T, 150]
    Output: Mean.npy and Std.npy with shape [150]
    """
    data_root = Path(f"./dataset/{dataset_name}")
    motion_dir = data_root / "new_joints"

    # Load file list
    all_file = data_root / "all.txt"
    if not all_file.exists():
        print(f"Error: {all_file} not found!")
        return

    with open(all_file, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]

    print(f"Calculating statistics for {len(file_list)} samples...")

    # Collect all motion data
    all_motions = []
    skipped = 0

    for fname in tqdm(file_list, desc="Loading motions"):
        motion_path = motion_dir / f"{fname}.npy"
        if motion_path.exists():
            try:
                motion = np.load(motion_path)  # Shape: [T, 150]
                if motion.ndim != 2 or motion.shape[1] != 150:
                    print(f"Warning: {fname} has shape {motion.shape}, expected [T, 150]")
                    skipped += 1
                    continue
                all_motions.append(motion)
            except Exception as e:
                print(f"Error loading {fname}: {e}")
                skipped += 1
        else:
            print(f"File not found: {motion_path}")
            skipped += 1

    if not all_motions:
        print("No valid motion files found!")
        return

    print(f"Loaded {len(all_motions)} valid samples (skipped {skipped})")

    # Concatenate all frames
    all_data = np.concatenate(all_motions, axis=0)  # [Total_frames, 150]
    print(f"Total frames: {all_data.shape[0]}")

    # Calculate statistics
    mean = np.mean(all_data, axis=0)  # [150]
    std = np.std(all_data, axis=0)    # [150]

    # Avoid division by zero
    std[std < 1e-6] = 1.0

    # Save
    mean_path = data_root / "Mean.npy"
    std_path = data_root / "Std.npy"

    np.save(mean_path, mean)
    np.save(std_path, std)

    print(f"\nStatistics saved:")
    print(f"   {mean_path}")
    print(f"   {std_path}")
    print(f"\nStatistics summary:")
    print(f"   Mean shape: {mean.shape}")
    print(f"   Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"   Std shape: {std.shape}")
    print(f"   Std range: [{std.min():.4f}, {std.max():.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='YOUTUBE_SIGN',
                        help='Dataset name')
    args = parser.parse_args()

    calculate_statistics(args.dataset)
