import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Base path of the directory
base_path = "/gpfs/data1/cmongp1/GEOGLAM/Output/FEWSNET/crop_t20"

# Get a list of all countries (directories in the base path)
countries = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

# Folder names (as seen in the image)
folder_names = ["chirps", "chirps_gefs", "cpc_tmax", "cpc_tmin", "esi_4wk", "ndvi", "nsidc_rootzone", "nsidc_surface"]

# Data collection
file_counts = []

for country in tqdm(countries, desc="Countries"):
    # Dynamically find the deepest subdirectory (replace 'admin_1/cr' with auto-detection)
    country_path = os.path.join(base_path, country)
    subdirs = [os.path.join(root, d) for root, dirs, _ in os.walk(country_path) for d in dirs if "cr" in d]
    if subdirs:
        folder_path_root = subdirs[0]  # Assuming there's only one relevant "cr" path
    else:
        folder_path_root = country_path  # If no "cr" directory exists, fallback to the base country directory

    counts = []
    for folder in tqdm(folder_names, desc="Variables", leave=False):
        folder_path = os.path.join(folder_path_root, folder)
        if os.path.exists(folder_path):
            counts.append(len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]))
        else:
            counts.append(0)
    file_counts.append(counts)

# Create DataFrame
df = pd.DataFrame(file_counts, index=countries, columns=folder_names)

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Number of Files in Subfolders by Country")
plt.xlabel("Folder Names")
plt.ylabel("Countries")
plt.tight_layout()
plt.savefig("count_files.png", dpi=300)
