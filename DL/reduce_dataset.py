import os
import random
import shutil

def reduce_dataset(base_path, train_limit=200, valid_limit=50):
    dataset_dir = os.path.join(base_path, "New Plant Diseases Dataset(Augmented)")
    
    for split in ["train", "valid"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} not found. Skipping.")
            continue
            
        limit = train_limit if split == "train" else valid_limit
        print(f"Processing {split} split (Target limit: {limit} per class)...")
        
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        
        for class_name in classes:
            class_path = os.path.join(split_dir, class_name)
            files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            
            if len(files) > limit:
                print(f"  Reducing {class_name}: {len(files)} -> {limit}")
                # Shuffle to keep a random subset
                random.shuffle(files)
                files_to_remove = files[limit:]
                
                for f in files_to_remove:
                    os.remove(os.path.join(class_path, f))
            else:
                print(f"  Skipping {class_name}: {len(files)} <= {limit}")

if __name__ == "__main__":
    # You can adjust these numbers based on how much you want to reduce
    TRAIN_LIMIT = 200 
    VALID_LIMIT = 50
    
    current_dir = os.getcwd()
    reduce_dataset(current_dir, train_limit=TRAIN_LIMIT, valid_limit=VALID_LIMIT)
    print("Done!")
