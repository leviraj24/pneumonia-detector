# src/retrain_from_feedback.py
import os
import shutil
import subprocess
import argparse
from datetime import datetime

def integrate_feedback(feedback_dir, data_dir):
    """
    Moves corrected images from the feedback directory to the main training data directory.
    """
    train_dir = os.path.join(data_dir, "train")
    if not os.path.exists(feedback_dir):
        print(f"⚠️ Feedback directory '{feedback_dir}' not found. Nothing to integrate.")
        return 0

    integrated_count = 0
    print("\n🔍 Checking for new feedback...")

    for class_name in os.listdir(feedback_dir):
        class_feedback_dir = os.path.join(feedback_dir, class_name)
        class_train_dir = os.path.join(train_dir, class_name)

        if not os.path.isdir(class_feedback_dir):
            continue
            
        if not os.path.exists(class_train_dir):
            print(f"Warning: Training directory for class '{class_name}' does not exist. Skipping.")
            continue

        for filename in os.listdir(class_feedback_dir):
            source_path = os.path.join(class_feedback_dir, filename)
            dest_path = os.path.join(class_train_dir, filename)

            if os.path.exists(dest_path):
                print(f"  - File '{filename}' already exists in training set. Skipping.")
                continue

            shutil.move(source_path, dest_path)
            print(f"  ✅ Integrated '{filename}' into '{class_name}' training set.")
            integrated_count += 1
            
    if integrated_count == 0:
        print("  - No new feedback files to integrate.")
    else:
        print(f"\n🎉 Successfully integrated {integrated_count} new images into the training set.")
        
    return integrated_count

def run_training(model_name="efficientnet_b0", epochs_ft=15):
    """
    Calls the train_advanced.py script using a subprocess.
    """
    print("\n🚀 Starting the re-training pipeline...")
    print("="*50)
    
    command = [
        "python",
        "src/train_advanced.py",
        "--model", model_name,
        "--epochs-ft", str(epochs_ft),
    ]
    
    try:
        subprocess.run(command, check=True)
        print("\n✅ Re-training complete! Your model has been updated with the new data.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ An error occurred during training: {e}")
    except FileNotFoundError:
        print("\n❌ Error: Could not find 'src/train_advanced.py'. Make sure you are in the project's root directory.")

def main():
    parser = argparse.ArgumentParser(description="Active Learning Re-training Pipeline")
    
    # --- UPDATED DEFAULT PATH ---
    parser.add_argument("--feedback-dir", default=os.path.join("data", "feedback"), help="Directory where corrected images are stored.")
    # ----------------------------

    parser.add_argument("--data-dir", default="data", help="Main data directory.")
    parser.add_argument("--model", default="efficientnet_b0", help="Model to train.")
    parser.add_argument("--epochs-ft", type=int, default=15, help="Number of fine-tuning epochs.")
    args = parser.parse_args()

    count = integrate_feedback(args.feedback_dir, args.data_dir)

    if count > 0:
        run_training(args.model, args.epochs_ft)
    else:
        print("\n🏁 Pipeline finished. No re-training was needed as there was no new data.")

if __name__ == "__main__":
    main()