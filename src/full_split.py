import h5py
import numpy as np
from pathlib import Path
import math
import torch
from tqdm import tqdm
import shutil

from data.dataset import Dataset
from data.episode import Episode
from data.segment import SegmentId
from data.dataset import MeleeHdf5Dataset

def combine_and_split_into_chunks(input_dir: Path, output_dir: Path, chunk_size: int = 1000):
    """Process multiple HDF5 files and split them into chunks of 1000 frames each"""
    print(f"Reading from directory: {input_dir}")
    print(f"Writing chunks to: {output_dir}")
    
    total_frames = 0
    chunk_idx = 0
    current_chunk_frames = 0
    current_output = None
    
    # Process each HDF5 file
    input_files = sorted(input_dir.glob("*.hdf5"))
    print(f"Found {len(input_files)} HDF5 files to process")
    
    for input_file in tqdm(input_files, desc="Processing HDF5 files"):
        with h5py.File(input_file, 'r') as f:
            # Count frames in this file
            file_frames = 0
            while f'frame_{file_frames}_x' in f:
                file_frames += 1
            
            print(f"\nProcessing {input_file.name} with {file_frames} frames")
            
            # Process each frame
            for frame_idx in range(file_frames):
                # Create new chunk file if needed
                if current_output is None:
                    output_file = output_dir / f"melee_{chunk_idx}.hdf5"
                    current_output = h5py.File(output_file, 'w')
                    print(f"Creating new chunk: {output_file}")
                
                # Copy frame data
                frame_data = f[f'frame_{frame_idx}_x'][:]
                p1_data = f[f'frame_{frame_idx}_p1_y'][:]
                p2_data = f[f'frame_{frame_idx}_p2_y'][:]
                
                current_output.create_dataset(f'frame_{current_chunk_frames}_x', data=frame_data)
                current_output.create_dataset(f'frame_{current_chunk_frames}_p1_y', data=p1_data)
                current_output.create_dataset(f'frame_{current_chunk_frames}_p2_y', data=p2_data)
                
                current_chunk_frames += 1
                total_frames += 1
                
                # Close chunk if it's full
                if current_chunk_frames >= chunk_size:
                    current_output.close()
                    current_output = None
                    current_chunk_frames = 0
                    chunk_idx += 1
    
    # Close final chunk if it exists
    if current_output is not None:
        current_output.close()
        print(f"\nDropping {current_chunk_frames} incomplete frames from final chunk")
        # Remove incomplete chunk
        if current_chunk_frames < chunk_size:
            (output_dir / f"melee_{chunk_idx}.hdf5").unlink()
            chunk_idx -= 1
    
    print(f"\nProcessed {total_frames} total frames into {chunk_idx + 1} complete chunks")
    return chunk_idx + 1

def process_melee_data(input_dir: str, output_dir: str, train_ratio=0.9):
    """Process multiple Melee HDF5 files into train/test Episodes"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Setup directories
    low_res_dir = output_dir / "low_res"
    full_res_dir = output_dir / "full_res"

    for d in [low_res_dir/"train", low_res_dir/"test", full_res_dir]:
        d.mkdir(exist_ok=True, parents=True)

    # Combine and split files into chunks
    num_chunks = combine_and_split_into_chunks(input_dir, full_res_dir)

    # Create datasets
    melee_dataset = MeleeHdf5Dataset(full_res_dir)
    train_dataset = Dataset(low_res_dir / "train", None)
    test_dataset = Dataset(low_res_dir / "test", None)

    # Get filenames with proper prefix
    filenames = [f"full_res/{x.name}" for x in sorted(
        full_res_dir.glob("*.hdf5"),
        key=lambda x: int(x.stem.split("_")[-1])
    )]
    num_episodes = len(filenames)
    num_train = int(num_episodes * train_ratio)
    indices = np.random.permutation(num_episodes)
    train_indices = indices[:num_train]
    test_indices = indices[num_train:]

    print(f"\nProcessing {num_episodes} episodes into train/test splits")

    # Process episodes
    for i, filename in tqdm(enumerate(filenames), desc="Creating Episodes"):
        # Get episode data
        episode = Episode(
            **{
                k: v
                for k, v in melee_dataset[SegmentId(filename, 0, 1000)].__dict__.items()
                if k not in ("mask_padding", "id")
            }
        )

        # Add file reference to info
        episode.info = {"original_file_id": filename}

        # Add to appropriate dataset
        dataset = train_dataset if i in train_indices else test_dataset
        dataset.add_episode(episode)

    # Save datasets
    train_dataset.save_to_default_path()
    test_dataset.save_to_default_path()

    print(f"\nDataset split complete:")
    print(f"Train episodes: {train_dataset.num_episodes}")
    print(f"Test episodes: {test_dataset.num_episodes}")

if __name__ == "__main__":
    input_dir = r"C:\replays_hdf5"
    output_dir = r"C:\full_data"
    process_melee_data(input_dir, output_dir)
