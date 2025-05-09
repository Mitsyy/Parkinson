import os

# Paths to your normal and parkinson folders
normal_path = r"C:\Users\Mitali Mishra\OneDrive\Desktop\parkinsons\parkinsons_dataset\normal"
parkinson_path = r"C:\Users\Mitali Mishra\OneDrive\Desktop\parkinsons\parkinsons_dataset\parkinson"

# List all images in each folder
normal_images = os.listdir(normal_path)
parkinson_images = os.listdir(parkinson_path)

# Display the count
print(f"Number of images in 'normal': {len(normal_images)}")
print(f"Number of images in 'parkinson': {len(parkinson_images)}")
