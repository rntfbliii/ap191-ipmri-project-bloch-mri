# Bloch MRI Simulation

This project simulates MRI signal contrast using voxel-wise spatially varying \(T_1\) and \(T_2\) relaxation times under a spin-echo sequence, based on the Bloch equations.

## Structure
- `src/`: All simulation and plotting code
- `data/`: Phantom T1/T2 maps
- `results/`: Output images
- `notebooks/`: Jupyter notebooks for exploration
- `paper/`: LaTeX paper for SPP

## How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the simulation pipeline:
   ```bash
   python src/simulate.py
   ```

## Authors
Renato III F. Bolo  
Aldrin James R. Garcia  
Mariane R. Madlangsakay  
Crisleo John II E. Martinito  
Katlyn Faye B. Nacague
