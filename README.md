# DAGGER_Imitation_Learning
## Intelligent Control (ECE_ΔK807) – Team Projects
### 4. Imitation Learning with DAGGER or Privileged Information 🕹️

### 🧠 Topics
- Imitation Learning
- Dataset Aggregation
- Partial Observability

### 📄 Description
Train a policy using demonstrations from an expert with full-state access, while the policy has only partial or noisy observations (e.g., images or low-dimensional features). Implement **DAGGER** to iteratively improve the learner. Compare to **Behavior Cloning** and analyze generalization under noise.

---

## 🛠️ Setup Instructions

### 1. Create a new conda environment
```bash
conda create -n mario python=3.8 -y
```

### 2. Activate the environment
```bash
conda activate mario
```

### 3. Install dependencies
Make sure you have **Microsoft Visual C++ 14.0+** installed:  
https://visualstudio.microsoft.com/visual-cpp-build-tools/

```bash
pip install -r requirements_mario.txt
```

or

```bash
pip install gym==0.23.1
pip install nes-py==8.1.8
pip install gym-super-mario-bros==7.4.0
pip install opencv-python
pip install scikit-learn
pip install matplotlib
pip install seaborn
pip install torch
```

### 4. (Optional) Install CUDA-compatible PyTorch
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---


## Τα βίντεο μας στο YouTube:

https://youtu.be/h9JXUzJS6rU?si=cJCze_dh0Qy8pmU7

https://youtu.be/ixdq4bWbvA0?si=uFaUeUJ6ANvZHXFK
