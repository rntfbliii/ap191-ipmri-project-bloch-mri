# ========================
# Imports
# ========================
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ========================
# Phantom Construction
# ========================
def create_phantom(matrix_size=128, lesion_radius=25,
                   T1_bg=920, T2_bg=100,
                   T1_core=1000, T1_edge=1400,
                   T2_core=90, T2_edge=130):
    T1_map = np.full((matrix_size, matrix_size), T1_bg, dtype=float)
    T2_map = np.full((matrix_size, matrix_size), T2_bg, dtype=float)

    X, Y = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))
    center = matrix_size // 2
    r = np.sqrt((X - center)**2 + (Y - center)**2)
    mask = r <= lesion_radius

    grad = r[mask] / lesion_radius
    T1_map[mask] = T1_core + (T1_edge - T1_core) * grad
    T2_map[mask] = T2_core + (T2_edge - T2_core) * grad

    return T1_map, T2_map

# ========================
# Signal Equation
# ========================
def spin_echo_signal(T1, T2, TR, TE):
    return (1 - np.exp(-TR / T1)) * np.exp(-TE / T2)

# ========================
# Display Utilities
# ========================
def display_maps(maps, titles, cmaps, labels):
    fig, axes = plt.subplots(1, len(maps), figsize=(6 * len(maps), 5))
    for ax, data, title, cmap, label in zip(axes, maps, titles, cmaps, labels):
        im = ax.imshow(data, cmap=cmap, origin='lower')
        ax.set_title(title)
        ax.axis('off')
        fig.colorbar(im, ax=ax, label=label)
    plt.tight_layout()
    plt.show()

def line_profile_plot(x, profiles, labels, colors, title):
    plt.figure(figsize=(10, 4))
    for y, label, color in zip(profiles, labels, colors):
        plt.plot(x, y, label=label, color=color)
    plt.title(title)
    plt.xlabel('Pixel Position (x)')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def moving_average(arr, window=5):
    return np.convolve(arr, np.ones(window) / window, mode='same')

def compute_LBCR(signal_map, lesion_radius):
    matrix_size = signal_map.shape[0]
    center = matrix_size // 2

    Y, X = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))
    r = np.sqrt((X - center)**2 + (Y - center)**2)

    lesion_mask = r <= lesion_radius
    background_mask = r >= lesion_radius + 10  # 10-pixel buffer

    lesion_signal = np.mean(signal_map[lesion_mask])
    background_signal = np.mean(signal_map[background_mask])

    lbcr = lesion_signal / background_signal
    return lbcr

# ========================
# Parameters
# ========================
matrix_size = 128
lesion_radius = 25
TR_default = 2000  # ms
TE_default = 100   # ms

T1_bg, T2_bg = 920, 100
T1_core, T1_edge = 1000, 1400
T2_core, T2_edge = 90, 130

# ========================
# Generate Phantom & Signal
# ========================
T1_map, T2_map = create_phantom(matrix_size, lesion_radius,
                                T1_bg, T2_bg, T1_core, T1_edge, T2_core, T2_edge)

signal_map = spin_echo_signal(T1_map, T2_map, TR_default, TE_default)
signal_map = (signal_map - np.min(signal_map)) / (np.max(signal_map) - np.min(signal_map))

# ========================
# Validation Print
# ========================
center = matrix_size // 2
print("Center T1, T2:", T1_map[center, center], T2_map[center, center])
print("Edge T1, T2:", T1_map[center, center + lesion_radius], T2_map[center, center + lesion_radius])

# ========================
# Plot Static Maps
# ========================
display_maps(
    [T1_map, T2_map, signal_map],
    ['$T_1$ Relaxation Map', '$T_2$ Relaxation Map', 'Spin Echo Signal'],
    ['viridis', 'plasma', 'gray'],
    ['$T_1$ (ms)', '$T_2$ (ms)', 'Signal Intensity']
)

# ========================
# TR/TE Grid Simulation
# ========================
TR_values = [500, 1000, 2000, 3000]
TE_values = [10, 50, 100, 150]

fig, axes = plt.subplots(len(TR_values), len(TE_values), figsize=(15, 15),
                         gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
fig.suptitle('MRI Signal Simulation (Spin-Echo Varying TR/TE)', y=1.02, fontsize=16)

for i, TR in enumerate(TR_values):
    for j, TE in enumerate(TE_values):
        ax = axes[i, j]
        signal = spin_echo_signal(T1_map, T2_map, TR, TE)
        im = ax.imshow(signal, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'TR={TR}ms\nTE={TE}ms', fontsize=9)
        ax.axis('off')
        if j == len(TE_values) - 1:
            cax = fig.add_axes([ax.get_position().x1 + 0.01,
                                ax.get_position().y0,
                                0.015,
                                ax.get_position().height])
            fig.colorbar(im, cax=cax, label='Signal Intensity')

plt.figtext(0.5, 0.01,
            'Note: Lesion at center with T1=1000–1400ms, T2=90–130ms. '
            'Background: T1=920ms, T2=100ms.',
            ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# ========================
# 1D Line Profiles
# ========================
row_idx = matrix_size // 2
x = np.arange(matrix_size)

plt.figure(figsize=(10, 4))
plt.plot(x, T1_map[row_idx], label='T1 (ms)', color='red')
plt.plot(x, T2_map[row_idx], label='T2 (ms)', color='blue')
plt.title('1D Line Profile: T1 and T2 Across Lesion (Row 64)')
plt.xlabel('Pixel Position (x)')
plt.ylabel('Relaxation Time (ms)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

smoothed_signal = moving_average(signal_map[row_idx], window=5)
plt.figure(figsize=(10, 4))
plt.plot(x, smoothed_signal, label='Signal Intensity (Smoothed)', color='black')
plt.title('1D Line Profile: Smoothed Signal Intensity Across Lesion (Row 64)')
plt.xlabel('Pixel Position (x)')
plt.ylabel('Signal Intensity (a.u.)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ========================
# TR/TE Comparison Visualization
# ========================
comparison_settings = [(2000, 100), (800, 30)]
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

for ax, (TR, TE) in zip(axs, comparison_settings):
    sig = spin_echo_signal(T1_map, T2_map, TR, TE)
    im = ax.imshow(sig, cmap='gray')
    ax.set_title(f'Signal (TR={TR}, TE={TE})')
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Signal Intensity (a.u.)')

plt.suptitle('TR/TE Comparison – Simulated MRI Contrast', fontsize=14)
plt.tight_layout()
plt.show()

# ========================
# TR/TE Grid + LBCR Overlay
# ========================
LBCR_matrix = np.zeros((len(TR_values), len(TE_values)))

fig, axes = plt.subplots(len(TR_values), len(TE_values), figsize=(15, 15),
                         gridspec_kw={'wspace': 0.4, 'hspace': 0.4})
fig.suptitle('MRI Signal & LBCR (Spin-Echo Varying TR/TE)', y=1.02, fontsize=16)

for i, TR in enumerate(TR_values):
    for j, TE in enumerate(TE_values):
        ax = axes[i, j]
        signal = spin_echo_signal(T1_map, T2_map, TR, TE)
        lbcr = compute_LBCR(signal, lesion_radius)
        LBCR_matrix[i, j] = lbcr

        im = ax.imshow(signal, cmap='gray', vmin=0, vmax=1)
        ax.set_title(f'TR={TR}ms\nTE={TE}ms\nLBCR={lbcr:.2f}', fontsize=9)
        ax.axis('off')

        if j == len(TE_values) - 1:
            cax = fig.add_axes([ax.get_position().x1 + 0.01,
                                ax.get_position().y0,
                                0.015,
                                ax.get_position().height])
            fig.colorbar(im, cax=cax, label='Signal Intensity')

plt.figtext(0.5, 0.01,
            'LBCR = Lesion-to-Background Contrast Ratio. Higher is better for visibility.',
            ha='center', fontsize=10)
plt.tight_layout()
plt.show()

# ========================
# Heatmap of LBCR(TR, TE)
# ========================
plt.figure(figsize=(8, 6))
c = plt.imshow(LBCR_matrix, cmap='hot', aspect='auto', origin='lower')
plt.xticks(range(len(TE_values)), TE_values)
plt.yticks(range(len(TR_values)), TR_values)
plt.xlabel('TE (ms)')
plt.ylabel('TR (ms)')
plt.title('LBCR Heatmap (TR vs TE)')
plt.colorbar(c, label='LBCR')
plt.tight_layout()
plt.show()

# ========================
# Surface Plot of LBCR
# ========================
TR_grid, TE_grid = np.meshgrid(TE_values, TR_values)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(TE_grid, TR_grid, LBCR_matrix, cmap=cm.viridis)
ax.set_xlabel('TE (ms)')
ax.set_ylabel('TR (ms)')
ax.set_zlabel('LBCR')
ax.set_title('LBCR Surface Plot')
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.tight_layout()
plt.show()

# ========================
# Dual 3D Surface Plot: T1 and T2 Maps
# ========================
X, Y = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))
downsample = 2
X_ds = X[::downsample, ::downsample]
Y_ds = Y[::downsample, ::downsample]
T1_ds = T1_map[::downsample, ::downsample]
T2_ds = T2_map[::downsample, ::downsample]

fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X_ds, Y_ds, T1_ds, cmap='plasma', edgecolor='none')
ax1.set_title('$T_1$ Relaxation Map')
ax1.set_xlabel('X (pixels)')
ax1.set_ylabel('Y (pixels)')
ax1.set_zlabel('$T_1$ (ms)')
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=10, pad=0.1, label='$T_1$ (ms)')

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X_ds, Y_ds, T2_ds, cmap='viridis', edgecolor='none')
ax2.set_title('$T_2$ Relaxation Map')
ax2.set_xlabel('X (pixels)')
ax2.set_ylabel('Y (pixels)')
ax2.set_zlabel('$T_2$ (ms)')
fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=10, pad=0.1, label='$T_2$ (ms)')

plt.suptitle("Dual-Surface Comparison of T₁ and T₂ Relaxation Maps", fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

# ========================
# LBCR vs TE and TR
# ========================
TR_values = np.linspace(500, 3000, LBCR_matrix.shape[0])
TE_values = np.linspace(10, 150, LBCR_matrix.shape[1])

fixed_TR_value = 2000
fixed_TR_idx = np.argmin(np.abs(TR_values - fixed_TR_value))
lbcr_vs_TE = LBCR_matrix[fixed_TR_idx, :]
max_TE_idx = np.argmax(lbcr_vs_TE)
max_TE = TE_values[max_TE_idx]
max_LBCR_TE = lbcr_vs_TE[max_TE_idx]

plt.figure(figsize=(8, 4))
plt.plot(TE_values, lbcr_vs_TE, marker='o', label=f'TR ≈ {TR_values[fixed_TR_idx]:.0f} ms')
plt.scatter(max_TE, max_LBCR_TE, color='red', zorder=5, label=f'Max LBCR @ TE={max_TE:.0f}ms')
plt.title('LBCR vs TE at Fixed TR')
plt.xlabel('TE (ms)')
plt.ylabel('LBCR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

fixed_TE_value = 100
fixed_TE_idx = np.argmin(np.abs(TE_values - fixed_TE_value))
lbcr_vs_TR = LBCR_matrix[:, fixed_TE_idx]
max_TR_idx = np.argmax(lbcr_vs_TR)
max_TR = TR_values[max_TR_idx]
max_LBCR_TR = lbcr_vs_TR[max_TR_idx]

plt.figure(figsize=(8, 4))
plt.plot(TR_values, lbcr_vs_TR, marker='s', color='orange', label=f'TE ≈ {TE_values[fixed_TE_idx]:.0f} ms')
plt.scatter(max_TR, max_LBCR_TR, color='red', zorder=5, label=f'Max LBCR @ TR={max_TR:.0f}ms')
plt.title('LBCR vs TR at Fixed TE')
plt.xlabel('TR (ms)')
plt.ylabel('LBCR')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
