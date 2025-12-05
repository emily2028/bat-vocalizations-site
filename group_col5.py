# categorize_audio.py
import os
import sys
import shutil
from tkinter import Tk, filedialog, simpledialog

# --- Dependency Checking ---
try:
    import librosa
except ImportError:
    print("Missing dependency: 'librosa' is required. Please install it with: pip install librosa")
    sys.exit(1)

try:
    from sklearn.cluster import KMeans
except ImportError:
    print("Missing dependency: 'scikit-learn' is required. Please install it with: pip install scikit-learn")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Missing dependency: 'numpy' is required. Please install it with: pip install numpy")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Missing dependency: 'matplotlib' is required. Please install it with: pip install matplotlib")
    sys.exit(1)

# --- User Interface Functions ---

def select_folders(title: str) -> list[str]:
    """Opens multiple folder dialogs until the user cancels."""
    root = Tk()
    root.withdraw()
    folders = []
    while True:
        folder = filedialog.askdirectory(title=f"{title} (select a folder, or cancel to finish)")
        if not folder:
            break
        folders.append(folder)
    root.destroy()
    return folders

def select_output_folder(title: str) -> str:
    """Opens a dialog to select a single output folder."""
    root = Tk()
    root.withdraw()
    folder = filedialog.askdirectory(title=title)
    root.destroy()
    return folder

# --- Core Logic ---

def extract_features(file_path: str) -> np.ndarray | None:
    """
    Extracts a rich set of acoustic features from an audio file.
    Returns a single feature vector representing the audio.
    """
    try:
        # Load audio file. 'sr=None' preserves the original sample rate.
        y, sr = librosa.load(file_path, sr=None)
        
        # --- Feature 1: MFCCs (timbre) ---
        # We'll capture both the mean and standard deviation to get dynamics.
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # --- Feature 2: Spectral Centroid (brightness) ---
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

        # --- Feature 3: Spectral Bandwidth (frequency range) ---
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

        # --- Feature 4: Zero-Crossing Rate (percussiveness/tonality) ---
        zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

        # --- Feature 5: Syllable Count (approximated by onset detection) ---
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        syllable_count = len(onsets)

        # --- Feature 6: Chroma Features (tonal characteristics) ---
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)

        # --- Feature 7: Spectral Contrast (peak/valley differences) ---
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast, axis=1)
        contrast_std = np.std(contrast, axis=1)

        # --- Feature 8: Pitch (Fundamental Frequency) ---
        # pyin provides fundamental frequency (f0) and voicing information
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        # Only consider voiced frames for pitch statistics
        voiced_f0 = f0[voiced_flag]
        pitch_mean = np.mean(voiced_f0) if len(voiced_f0) > 0 else 0
        pitch_std = np.std(voiced_f0) if len(voiced_f0) > 0 else 0

        # --- Combine all features into a single vector ---
        feature_vector = np.hstack((
            mfccs_mean, mfccs_std, spectral_centroid, spectral_bandwidth, zcr, syllable_count,
            chroma_mean, chroma_std,
            contrast_mean, contrast_std,
            pitch_mean, pitch_std
        ))
        return feature_vector
    except Exception as e:
        print(f"  - Warning: Could not process file {os.path.basename(file_path)}. Error: {e}")
        return None

def categorize_vocalizations(input_folders: list[str], output_folder: str, n_clusters: int):
    """
    Finds all WAV files, extracts features, clusters them, and copies them
    into categorized subfolders.
    """
    # 1. Find all .wav files
    print("Scanning for .wav files...")
    wav_files = []
    for folder in input_folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.lower().endswith('.wav'):
                    wav_files.append(os.path.join(root, file))

    if not wav_files:
        print("No .wav files found in the selected folders. Exiting.")
        return

    print(f"Found {len(wav_files)} .wav files. Extracting features...")

    # 2. Extract features from each file
    feature_list = []
    valid_files = []
    for i, file_path in enumerate(wav_files):
        print(f"  ({i+1}/{len(wav_files)}) Processing {os.path.basename(file_path)}...")
        features = extract_features(file_path)
        if features is not None:
            feature_list.append(features)
            valid_files.append(file_path)

    if len(feature_list) < n_clusters:
        print(f"Error: The number of successfully processed files ({len(feature_list)}) is less than the requested number of groups ({n_clusters}).")
        print("Cannot proceed with clustering.")
        return

    # 3. Perform K-Means clustering
    print(f"\nClustering files into {n_clusters} groups based on acoustic features...")
    X = np.array(feature_list)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto').fit(X)
    labels = kmeans.labels_
    print("Clustering complete.")

    # 4. Create output directories and copy files
    print("\nCopying files into categorized folders...")
    for i in range(n_clusters):
        os.makedirs(os.path.join(output_folder, f"group_{i}"), exist_ok=True)

    for i, file_path in enumerate(valid_files):
        label = labels[i]
        dest_folder = os.path.join(output_folder, f"group_{label}")
        shutil.copy(file_path, dest_folder)
        print(f"  - Copied '{os.path.basename(file_path)}' to 'group_{label}'")

    print(f"\nProcess complete. Files have been categorized into {n_clusters} groups inside '{output_folder}'.")


if __name__ == '__main__':
    # 1. Get user input
    input_dirs = select_folders("Select an input folder with WAV files")
    if not input_dirs:
        print("No input folders selected. Exiting.")
        sys.exit()

    output_dir = select_output_folder("Select a base folder to save the categorized groups")
    if not output_dir:
        print("No output folder selected. Exiting.")
        sys.exit()

    # --- Configuration ---
    NUM_GROUPS = 28

    # --- Run the main process ---
    categorize_vocalizations(
        input_folders=input_dirs,
        output_folder=output_dir,
        n_clusters=NUM_GROUPS
    )
