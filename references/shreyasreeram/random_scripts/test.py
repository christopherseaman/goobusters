import torch

checkpoint_path = "/Users/Shreya1/tools/sea_raft/models/Tartan-C-T-TSKH-spring540x960-M.pth"

# Load checkpoint
checkpoint = torch.load(checkpoint_path, map_location="cpu")

# Check if it contains "state_dict"
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

# Print all layer names and their dimensions
print("\nCheckpoint layers and shapes:")
for key, value in checkpoint.items():
    print(f"{key}: {value.shape}")