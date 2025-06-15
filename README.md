# DAGGER_Imitation_Learning
## Intelligent Control – Team Projects 
### 4. Imitation Learning with DAGGER or Privileged Information

**Topics:** Imitation Learning, Dataset Aggregation, Partial Observability

**Description:** Train a policy using demonstrations from an expert with full-state access,
while the policy has only partial or noisy observations (e.g., images or low-dimensional features). Implement DAGGER to iteratively improve the learner. Compare to behavior cloning
and analyze generalization under noise.

# --- Create a new conda environment ---
conda create -n mario python=3.8 -y

# --- Activate the environment ---
conda activate mario

# --- Install required Python packages via pip ---
Microsoft Visual C++ 14.0 or greater is required.
https://visualstudio.microsoft.com/visual-cpp-build-tools/

pip install -r requirements_mario.txt

or

pip install gym==0.23.1
pip install nes-py==8.1.8
pip install gym-super-mario-bros==7.4.0
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install torch

# --- For CUDA-compatible version of PyTorch ---
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
