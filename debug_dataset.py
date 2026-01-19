from datasets import get_dataset_config_names
try:
    configs = get_dataset_config_names("mythicinfinity/libritts")
    print("Available configs:", configs)
except Exception as e:
    print(f"Error getting configs: {e}")
