import h5py
import numpy as np
import argparse
import re

def process_frame_data(file_path):
    """
    Process HDF5 file with frame-by-frame coordinate data.
    
    Parameters:
    file_path (str): Path to the HDF5 file
    
    Returns:
    dict: Organized frame data
    """
    with h5py.File(file_path, 'r') as f:
        # Get all keys
        keys = list(f.keys())
        print("Number of entries:", len(keys))
        
        # Extract frame numbers and create organized structure
        frame_data = {}
        frame_pattern = re.compile(r'frame_(\d+)_(.+)')
        
        for key in keys:
            match = frame_pattern.match(key)
            if match:
                frame_num = int(match.group(1))
                data_type = match.group(2)  # x, p1_y, or p2_y
                
                # Initialize frame entry if it doesn't exist
                if frame_num not in frame_data:
                    frame_data[frame_num] = {}
                
                # Store the data
                frame_data[frame_num][data_type] = np.array(f[key])
        
        # Convert to numpy arrays
        frame_numbers = sorted(frame_data.keys())
        x_coords = np.array([frame_data[f]['x'] for f in frame_numbers if 'x' in frame_data[f]])
        p1_y_coords = np.array([frame_data[f]['p1_y'] for f in frame_numbers if 'p1_y' in frame_data[f]])
        p2_y_coords = np.array([frame_data[f]['p2_y'] for f in frame_numbers if 'p2_y' in frame_data[f]])
        
        # Print statistics
        print(f"\nTotal frames processed: {len(frame_numbers)}")
        print("\nStatistics:")
        print("X coordinates shape:", x_coords.shape)
        print("P1 Y coordinates shape:", p1_y_coords.shape)
        print("P2 Y coordinates shape:", p2_y_coords.shape)
        
        print("\nX coordinates stats:")
        print("Mean:", np.mean(x_coords))
        print("Std:", np.std(x_coords))
        
        print("\nP1 Y coordinates stats:")
        print("Mean:", np.mean(p1_y_coords))
        print("Std:", np.std(p1_y_coords))
        
        print("\nP2 Y coordinates stats:")
        print("Mean:", np.mean(p2_y_coords))
        print("Std:", np.std(p2_y_coords))
        
        return {
            'frame_numbers': np.array(frame_numbers),
            'x_coords': x_coords,
            'p1_y_coords': p1_y_coords,
            'p2_y_coords': p2_y_coords,
            'raw_frame_data': frame_data
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process HDF5 file containing frame coordinate data.')
    parser.add_argument('file_path', type=str, help='Path to the HDF5 file')
    parser.add_argument('--save', action='store_true', help='Save the processed data as numpy files')
    
    args = parser.parse_args()
    
    try:
        print(f"Processing file: {args.file_path}")
        data = process_frame_data(args.file_path)
        
        if args.save:
            np.save('frame_numbers.npy', data['frame_numbers'])
            np.save('x_coords.npy', data['x_coords'])
            np.save('p1_y_coords.npy', data['p1_y_coords'])
            np.save('p2_y_coords.npy', data['p2_y_coords'])
            print("\nSaved data to:")
            print("- frame_numbers.npy")
            print("- x_coords.npy")
            print("- p1_y_coords.npy")
            print("- p2_y_coords.npy")
            
    except FileNotFoundError:
        print(f"Error: File '{args.file_path}' not found")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
