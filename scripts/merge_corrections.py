import os
import shutil
import json
from datetime import datetime
from PIL import Image
import hashlib

def validate_image(image_path):
    """
    Validate if an image file is valid and readable
    
    Args:
        image_path: Path to the image file
        
    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        with Image.open(image_path) as img:
            img.verify()  # Verify the image
        
        # Additional checks
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            return False, "Empty file"
        
        if file_size < 1024:  # Less than 1KB
            return False, "File too small, possibly corrupted"
            
        return True, None
        
    except Exception as e:
        return False, str(e)

def find_duplicate_images(directory):
    """
    Find duplicate images in a directory using MD5 hashing
    
    Args:
        directory: Directory to scan for duplicates
        
    Returns:
        dict: Dictionary mapping hash to list of file paths
    """
    hash_dict = {}
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash not in hash_dict:
                        hash_dict[file_hash] = []
                    hash_dict[file_hash].append(file_path)
                    
                except Exception as e:
                    print(f"⚠️ Could not hash {file_path}: {e}")
    
    # Filter to only duplicates
    duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}
    return duplicates

def clean_dataset(data_dir, backup_dir="data_backup", fix_corrupted=True, remove_duplicates=True):
    """
    Clean and validate the dataset
    
    Args:
        data_dir: Root directory of the dataset
        backup_dir: Directory to backup original data
        fix_corrupted: Whether to remove corrupted images
        remove_duplicates: Whether to remove duplicate images
        
    Returns:
        dict: Report of cleaning operations
    """
    print("🧹 Starting dataset cleaning...")
    
    # Create backup
    if not os.path.exists(backup_dir):
        print(f"📁 Creating backup at {backup_dir}...")
        shutil.copytree(data_dir, backup_dir)
        print("✅ Backup created")
    
    report = {
        'total_files': 0,
        'valid_files': 0,
        'corrupted_files': [],
        'removed_duplicates': [],
        'errors': []
    }
    
    # Scan all images
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                file_path = os.path.join(root, file)
                report['total_files'] += 1
                
                # Validate image
                is_valid, error_msg = validate_image(file_path)
                
                if is_valid:
                    report['valid_files'] += 1
                else:
                    report['corrupted_files'].append({
                        'path': file_path,
                        'error': error_msg
                    })
                    
                    if fix_corrupted:
                        try:
                            os.remove(file_path)
                            print(f"🗑️ Removed corrupted: {file_path}")
                        except Exception as e:
                            report['errors'].append(f"Could not remove {file_path}: {e}")
    
    # Find and remove duplicates
    if remove_duplicates:
        print("🔍 Scanning for duplicates...")
        duplicates = find_duplicate_images(data_dir)
        
        for file_hash, file_list in duplicates.items():
            if len(file_list) > 1:
                # Keep the first file, remove others
                keep_file = file_list[0]
                for duplicate_file in file_list[1:]:
                    try:
                        os.remove(duplicate_file)
                        report['removed_duplicates'].append(duplicate_file)
                        print(f"🗑️ Removed duplicate: {duplicate_file}")
                    except Exception as e:
                        report['errors'].append(f"Could not remove duplicate {duplicate_file}: {e}")
    
    return report

def reorganize_dataset(source_dir, target_dir, train_split=0.7, val_split=0.2, test_split=0.1):
    """
    Reorganize dataset into train/val/test splits
    
    Args:
        source_dir: Source directory with class folders
        target_dir: Target directory for reorganized data
        train_split: Fraction of data for training
        val_split: Fraction of data for validation  
        test_split: Fraction of data for testing
    """
    import random
    
    # Validate splits
    if abs(train_split + val_split + test_split - 1.0) > 0.01:
        raise ValueError("Splits must sum to 1.0")
    
    os.makedirs(target_dir, exist_ok=True)
    
    # Get all classes
    classes = [d for d in os.listdir(source_dir) 
              if os.path.isdir(os.path.join(source_dir, d))]
    
    print(f"📂 Found classes: {classes}")
    
    for class_name in classes:
        class_path = os.path.join(source_dir, class_name)
        
        # Get all images in this class
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split indices
        n_total = len(images)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        # Split images
        train_images = images[:n_train]
        val_images = images[n_train:n_train + n_val]
        test_images = images[n_train + n_val:]
        
        # Create directories and copy files
        for split, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            if split_images:  # Only create if there are images
                split_class_dir = os.path.join(target_dir, split, class_name)
                os.makedirs(split_class_dir, exist_ok=True)
                
                for image in split_images:
                    src_path = os.path.join(class_path, image)
                    dst_path = os.path.join(split_class_dir, image)
                    shutil.copy2(src_path, dst_path)
        
        print(f"📊 {class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")

def fix_file_extensions(data_dir):
    """
    Fix common file extension issues
    
    Args:
        data_dir: Directory to scan and fix
    """
    fixes = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Try to determine actual file type
            try:
                with Image.open(file_path) as img:
                    actual_format = img.format.lower()
                    
                # Get current extension
                current_ext = os.path.splitext(file)[1].lower()
                
                # Map formats to extensions
                format_to_ext = {
                    'jpeg': '.jpg',
                    'png': '.png',
                    'bmp': '.bmp',
                    'tiff': '.tiff'
                }
                
                correct_ext = format_to_ext.get(actual_format, current_ext)
                
                # Fix if needed
                if current_ext != correct_ext:
                    new_path = os.path.splitext(file_path)[0] + correct_ext
                    os.rename(file_path, new_path)
                    fixes.append((file_path, new_path))
                    print(f"🔧 Fixed: {file} -> {os.path.basename(new_path)}")
                    
            except Exception:
                continue  # Skip non-image files
    
    return fixes

def create_cleaning_report(report, save_path="reports/cleaning_report.json"):
    """Save dataset cleaning report"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Add timestamp
    report['cleaning_timestamp'] = datetime.now().isoformat()
    
    with open(save_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Cleaning report saved to: {save_path}")

def merge_dataset_folders(source_dirs, target_dir, class_mapping=None):
    """
    Merge multiple dataset folders into one organized structure
    
    Args:
        source_dirs: List of source directories to merge
        target_dir: Target directory for merged dataset
        class_mapping: Optional dictionary to map old class names to new ones
    """
    if class_mapping is None:
        class_mapping = {}
    
    os.makedirs(target_dir, exist_ok=True)
    
    merged_count = 0
    
    for source_dir in source_dirs:
        if not os.path.exists(source_dir):
            print(f"⚠️ Source directory not found: {source_dir}")
            continue
            
        print(f"📂 Merging from: {source_dir}")
        
        for item in os.listdir(source_dir):
            item_path = os.path.join(source_dir, item)
            
            if os.path.isdir(item_path):
                # Map class name if needed
                class_name = class_mapping.get(item, item)
                target_class_dir = os.path.join(target_dir, class_name)
                os.makedirs(target_class_dir, exist_ok=True)
                
                # Copy all images from this class
                for file in os.listdir(item_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                        src_file = os.path.join(item_path, file)
                        
                        # Create unique filename to avoid conflicts
                        base_name, ext = os.path.splitext(file)
                        unique_name = f"{os.path.basename(source_dir)}_{base_name}{ext}"
                        dst_file = os.path.join(target_class_dir, unique_name)
                        
                        # Copy file
                        shutil.copy2(src_file, dst_file)
                        merged_count += 1
    
    print(f"✅ Merged {merged_count} images into {target_dir}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset cleaning and correction utilities")
    parser.add_argument("--clean", action="store_true", help="Clean the dataset")
    parser.add_argument("--reorganize", action="store_true", help="Reorganize dataset into train/val/test")
    parser.add_argument("--fix-extensions", action="store_true", help="Fix file extensions")
    parser.add_argument("--data-dir", default="data", help="Data directory path")
    
    args = parser.parse_args()
    
    if args.clean:
        print("🧹 Cleaning dataset...")
        report = clean_dataset(args.data_dir)
        
        print(f"\n📊 Cleaning Summary:")
        print(f"  Total files processed: {report['total_files']}")
        print(f"  Valid files: {report['valid_files']}")
        print(f"  Corrupted files removed: {len(report['corrupted_files'])}")
        print(f"  Duplicates removed: {len(report['removed_duplicates'])}")
        
        if report['errors']:
            print(f"  Errors: {len(report['errors'])}")
            for error in report['errors']:
                print(f"    - {error}")
        
        create_cleaning_report(report)
    
    if args.fix_extensions:
        print("🔧 Fixing file extensions...")
        fixes = fix_file_extensions(args.data_dir)
        print(f"✅ Fixed {len(fixes)} file extensions")
    
    if args.reorganize:
        print("📂 Reorganizing dataset...")
        backup_dir = f"{args.data_dir}_original"
        reorganize_dataset(args.data_dir, f"{args.data_dir}_reorganized")
        print("✅ Dataset reorganized")
    
    if not any([args.clean, args.reorganize, args.fix_extensions]):
        print("Usage examples:")
        print("  python merge_corrections.py --clean")
        print("  python merge_corrections.py --fix-extensions")
        print("  python merge_corrections.py --reorganize")
        print("  python merge_corrections.py --clean --fix-extensions")