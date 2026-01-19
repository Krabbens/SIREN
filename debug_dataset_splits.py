from datasets import get_dataset_split_names
try:
    splits = get_dataset_split_names("mythicinfinity/libritts", "clean")
    print("Splits for 'clean':", splits)
except Exception as e:
    print(f"Error getting splits: {e}")
