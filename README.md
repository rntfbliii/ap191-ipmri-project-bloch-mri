# Bloch MRI Simulation

This project simulates MRI signal contrast by modeling voxel-wise spatial variations in \(T_1\) and \(T_2\) relaxation times within a synthetic phantom, under a spin-echo pulse sequence governed by the Bloch equations. It systematically explores \(T_R\)/\(T_E\) parameter sweeps and evaluates contrast behavior using quantitative metrics such as the Lesion-to-Background Contrast Ratio (LBCR) and Signal Difference (LBSD).

## Structure
- `src/`: The complete simulation and plotting code
- `paper/`: LaTeX paper and associated files
- `paper/figures`: Generated plots, graphs, and heatmaps

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation pipeline:
   ```bash
   python src/full_code.py
   ```

## Authors
Renato III F. Bolo  
Aldrin James R. Garcia  
Mariane R. Madlangsakay  
Crisleo John II E. Martinito  
Katlyn Faye B. Nacague
