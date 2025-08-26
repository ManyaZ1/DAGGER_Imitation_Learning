# DropoutDAgger Implementation

This directory contains the implementation of DropoutDAgger, an extension of the DAgger (Dataset Aggregation) algorithm that incorporates dropout regularization for improved generalization and robustness.

## What is DropoutDAgger?

DropoutDAgger extends the standard DAgger algorithm by adding dropout regularization to the neural network. This helps prevent overfitting and improves the agent's ability to generalize to new situations.

### Key Features:

1. **Dropout Regularization**: Added dropout layers to the DQN network to prevent overfitting
2. **Multiple Observation Modes**: Supports all the same observation modes as regular DAgger:
   - Full state observations
   - Partial observations (2/4 channels)
   - Noisy observations (σ=0.1, σ=0.2)
   - Downsampled images
3. **Configurable Dropout Rate**: Adjustable dropout rate (default 0.5)
4. **Training and Testing Infrastructure**: Complete training and evaluation pipeline

## Files Overview

### Core Implementation
- `dropout_dagger_agent.py` - DropoutDAgger agent with dropout-enabled DQN
- `dropout_dagger_trainer.py` - Training infrastructure for DropoutDAgger
- `dropout_dagger_main.py` - Main training script for all observation modes

### Testing and Evaluation
- `dropout_dagger_test.py` - Basic testing script
- `dropout_dagger_test_interactive.py` - Interactive testing with visualization
- `evaluate_dropout_dagger_agents.py` - Compare different DropoutDAgger models
- `test_dropout_dagger_implementation.py` - Implementation validation script
- `validate_dropout_core.py` - Core component validation

## Architecture

### DropoutDQN Network
The `DropoutDQN` class extends the standard DQN with:
- Dropout2d layers in convolutional layers (lighter dropout: dropout_rate * 0.25)
- Dropout layer in fully connected layers (full dropout_rate)
- Automatic train/eval mode switching for proper dropout behavior

### DropoutDaggerMarioAgent
Inherits from `DaggerMarioAgent` and:
- Uses DropoutDQN networks instead of regular DQN
- Properly handles dropout during training and inference
- Saves/loads dropout rate configuration
- Maintains all DAgger functionality

### DropoutDaggerTrainer
Provides complete training infrastructure:
- Support for all observation modes (partial, noisy, downsampled, full)
- Configurable dropout rates
- Comprehensive metrics tracking
- Model checkpointing and evaluation

## Usage

### Training DropoutDAgger Models

```bash
# Interactive training menu
python dropout_dagger_main.py

# This will show options for:
# 1. Full State
# 2. Partial Observations (2/4 channels)  
# 3. Noisy Observations (σ=0.1)
# 4. Noisy Observations (σ=0.2)
# 5. Downsampled Images
# 6. Train All
```

### Testing Trained Models

```bash
# Interactive testing
python dropout_dagger_test_interactive.py

# Basic testing (update model path first)
python dropout_dagger_test.py
```

### Evaluating Multiple Models

```bash
# Compare all DropoutDAgger variants
python evaluate_dropout_dagger_agents.py
```

### Validating Implementation

```bash
# Test core components without environment
python validate_dropout_core.py

# Full implementation test (may fail on environment setup)
python test_dropout_dagger_implementation.py
```

## Configuration

### DropoutDaggerConfig Parameters

```python
@dataclass
class DropoutDaggerConfig:
    iterations: int                    # Number of DAgger iterations
    episodes_per_iter: int            # Episodes per iteration  
    training_batches_per_iter: int    # Training batches per iteration
    expert_model_path: str            # Path to expert model
    dropout_rate: float = 0.5         # Dropout rate (NEW)
    observation_type: str = None      # 'partial', 'noisy', 'downsampled', or None
    noise_level: float = 0.1          # For noisy observations
    world: str = '1'                  # Mario world
    stage: str = '1'                  # Mario stage  
    render: bool = False              # Render during training
    save_frequency: int = 1           # Model save frequency
    max_episode_steps: int = 1000     # Max steps per episode
    only_for_testing: bool = False    # Testing mode only
```

## Observation Modes

### 1. Full State
- Agent sees the complete 4-channel stacked frames
- No observation wrapper applied
- Baseline configuration

### 2. Partial Observations  
- Agent only sees 2 out of 4 channels (most recent frames)
- Tests ability to work with limited temporal information
- Uses `PartialObservationWrapper('partial')`

### 3. Noisy Observations
- Gaussian noise added to observations
- Two noise levels: σ=0.1 (light) and σ=0.2 (heavy)
- Tests robustness to sensor noise
- Uses `PartialObservationWrapper('noisy', noise_level=X)`

### 4. Downsampled Images
- Images downsampled to 42x42 then upsampled back to 84x84
- Tests ability to work with lower resolution inputs
- Uses `PartialObservationWrapper('downsampled')`

## Results Structure

Trained models and results are saved in:
```
dropout_dagger_results/
├── experiment_YYYYMMDD_HHMMSS_<obs_type>_dropout<rate>/
│   ├── models/
│   ├── plots/
│   └── metrics/
```

## Comparison with Regular DAgger

| Feature | Regular DAgger | DropoutDAgger |
|---------|---------------|---------------|
| Network | Standard DQN | DropoutDQN |
| Regularization | None | Dropout layers |
| Overfitting | May overfit | Reduced overfitting |
| Generalization | Standard | Improved |
| Training Time | Faster | Slightly slower |
| Configuration | Standard params | + dropout_rate |

## Known Issues

1. **Environment Setup**: May encounter numpy compatibility issues with gym
2. **Performance**: Dropout adds slight computational overhead
3. **Hyperparameters**: Dropout rate may need tuning for different scenarios

## Future Improvements

1. **Adaptive Dropout**: Dynamic dropout rate adjustment during training
2. **Batch Normalization**: Combine with other regularization techniques  
3. **Ensemble Methods**: Multiple dropout models for uncertainty estimation
4. **Performance Analysis**: Detailed comparison with standard DAgger

## Requirements

Same as the main project:
- gym==0.23.1
- nes-py==8.1.8  
- gym-super-mario-bros==7.4.0
- opencv-python
- scikit-learn
- matplotlib
- seaborn
- torch

## Citation

If you use this DropoutDAgger implementation, please cite the original DAgger paper and mention this extension:

```
Ross, S., Gordon, G., & Bagnell, D. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. AISTATS.
```