# filter_audio.py
import os
import math
from tkinter import Tk, filedialog
try:
    from pydub import AudioSegment
    from pydub.silence import split_on_silence
except ImportError:
    import sys
    print("Missing dependency: the 'pydub' package is required. Install it with: pip install pydub")
    sys.exit(1)

def select_file(title: str) -> str:
    """Opens a file dialog to select a file."""
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path

def select_folder(title: str) -> str:
    """Opens a folder dialog to select a directory."""
    root = Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path

def process_audio(file_path: str, output_folder: str, min_silence_len: int = 500, silence_thresh_db: int = -40):
    """
    Splits an audio file by silence, saves all chunks, and then removes the quiet ones.

    Args:
        file_path (str): Path to the input WAV file.
        output_folder (str): Path to the folder to save chunk files.
        min_silence_len (int): The minimum duration of silence (in ms) to be
                               considered a separator between clusters.
        silence_thresh_db (int): The volume (in dBFS) below which is considered silence.
    """
    print(f"Loading audio file: {file_path}...")
    try:
        # Detect format and load
        file_extension = os.path.splitext(file_path)[1].lower().replace('.', '')
        if not file_extension:
            file_extension = "wav" # Assume wav if no extension
        audio = AudioSegment.from_file(file_path, format=file_extension)
    except Exception as e:
        print(f"Error loading audio file: {e}")
        return
    
    print(f"Splitting audio on silence (min duration: {min_silence_len}ms, threshold: {silence_thresh_db}dBFS)...")

    # Split the audio into chunks based on silence
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,  # ms
        silence_thresh=silence_thresh_db,
        keep_silence=100  # Keep a bit of silence at the start/end of chunks
    )

    if not chunks:
        print("No sound clusters found based on the silence settings. Exiting.")
        return

    print(f"Found {len(chunks)} sound clusters. Saving all to '{output_folder}'...")
    os.makedirs(output_folder, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(file_path))[0]

    saved_files = []
    for i, chunk in enumerate(chunks):
        chunk_filename = f"{base_filename}_chunk_{i:03d}.wav"
        chunk_path = os.path.join(output_folder, chunk_filename)
        chunk.export(chunk_path, format="wav")
        saved_files.append({'path': chunk_path, 'volume': chunk.dBFS})
        print(f"  - Saved {chunk_filename} (Avg Volume: {chunk.dBFS:.2f} dBFS)")

    print("\nFiltering saved chunks...")

    # Find the chunk with the highest average volume
    if not saved_files:
        print("No files were saved. Cannot perform filtering.")
        return
        
    loudest_chunk = max(saved_files, key=lambda x: x['volume'])
    loudest_volume_db = loudest_chunk['volume']
    print(f"Loudest chunk is '{os.path.basename(loudest_chunk['path'])}' with volume: {loudest_volume_db:.2f} dBFS.")

    # Calculate the threshold: 30% of linear amplitude is ~10.45 dB lower
    # Formula: 20 * log10(0.30) ≈ -10.45
    volume_threshold_db = loudest_volume_db - 10.45
    print(f"Volume threshold for keeping files: > {volume_threshold_db:.2f} dBFS (30% of loudest).")
    
    # Iterate through saved files and delete the ones that are too quiet
    deleted_count = 0
    for file_info in saved_files:
        if file_info['volume'] < volume_threshold_db:
            try:
                os.remove(file_info['path'])
                print(f"  - Deleting '{os.path.basename(file_info['path'])}' (Volume: {file_info['volume']:.2f} dBFS)")
                deleted_count += 1
            except OSError as e:
                print(f"  - Error deleting file {file_info['path']}: {e}")
        else:
            print(f"  - Keeping '{os.path.basename(file_info['path'])}' (Volume: {file_info['volume']:.2f} dBFS)")
            
    print(f"\nProcess complete. Kept {len(saved_files) - deleted_count} files, deleted {deleted_count} files.")


if __name__ == '__main__':
    # 1. Ask user for input file
    input_file = select_file("Select an audio file to process")
    if not input_file:
        print("No input file selected. Exiting.")
    else:
        # 2. Ask user for output folder
        output_dir = select_folder("Select a folder to save the split audio chunks")
        if not output_dir:
            print("No output folder selected. Exiting.")
        else:
            # --- Configuration for splitting ---
            # The minimum duration of silence (in ms) to count as a split point.
            MIN_SILENCE_DURATION_MS = 700
            # The volume (in dBFS) below which is considered silence.
            # -40 is a good starting point for voice.
            SILENCE_THRESHOLD_DBFS = -40
    
            # 3. Run the processing function
            process_audio(
                file_path=input_file,
                output_folder=output_dir,
                min_silence_len=MIN_SILENCE_DURATION_MS,
                silence_thresh_db=SILENCE_THRESHOLD_DBFS
            )
