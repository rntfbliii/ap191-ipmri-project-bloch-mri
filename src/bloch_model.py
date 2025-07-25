import numpy as np
import matplotlib.pyplot as plt

# Parameters
# Phantom dimensions
matrix_size = 128
voxel_size = 1.0 
image_dims = (matrix_size, matrix_size)

# Spin-Echo Sequence Parameters (ms)
TR = 2000 
TE = 100  
proton_density = 1.0

# Tissue Relaxation Times
# Background (Gray Matter) (ms)
T1_bg = 920 
T2_bg = 100 

# Lesion Core (ms)
T1_core = 1000
T2_core = 90 

# Lesion Edge
T1_edge = 1400 
T2_edge = 130 

# Lesion properties
lesion_radius = 25 
center_x, center_y = matrix_size // 2, matrix_size // 2

# Phantom Creation although I think sa second person na to but mukha kaseng needed na rin gawin dito idk sorry po
# Initialize T1 and T2 maps with background values
T1_map = np.full(image_dims, T1_bg, dtype=float)
T2_map = np.full(image_dims, T2_bg, dtype=float)

# Create a grid for calculating distances
x = np.arange(matrix_size)
y = np.arange(matrix_size)
X, Y = np.meshgrid(x, y)

# Calculate Euclidean distance from the lesion center for each voxel
r = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

# Identify voxels within the lesion
lesion_mask = r <= lesion_radius

# Ensure r_max is not zero for division, it's the lesion_radius
r_max = lesion_radius

T1_map[lesion_mask] = T1_core + (T1_edge - T1_core) * (r[lesion_mask] / r_max)
T2_map[lesion_mask] = T2_core + (T2_edge - T2_core) * (r[lesion_mask] / r_max)

# MRI Signal Intensity Map
signal_map = proton_density * (1 - np.exp(-TR / T1_map)) * np.exp(-TE / T2_map)

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# T1 Map
im1 = axes[0].imshow(T1_map, cmap='viridis', origin='lower')
axes[0].set_title('$T_1$ Relaxation Map (ms)')
axes[0].axis('off')
fig.colorbar(im1, ax=axes[0], orientation='vertical', label='$T_1$ (ms)')

# T2 Map
im2 = axes[1].imshow(T2_map, cmap='plasma', origin='lower')
axes[1].set_title('$T_2$ Relaxation Map (ms)')
axes[1].axis('off')
fig.colorbar(im2, ax=axes[1], orientation='vertical', label='$T_2$ (ms)')

# Signal Intensity Map
im3 = axes[2].imshow(signal_map, cmap='gray', origin='lower') # 'gray' or 'hot' for signal intensity
axes[2].set_title('MRI Signal Intensity Map')
axes[2].axis('off')
fig.colorbar(im3, ax=axes[2], orientation='vertical', label='Signal Intensity')

plt.tight_layout()
plt.show()
