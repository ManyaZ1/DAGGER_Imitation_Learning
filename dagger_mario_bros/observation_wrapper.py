import numpy as np

class PartialObservationWrapper:
    """Wrapper για διαφορετικά observation scenarios"""
    
    def __init__(self, obs_type: str, noise_level: float = 0.1):
        self.obs_type = obs_type
        self.noise_level = noise_level
        
    def transform_observation(self, state: np.ndarray) -> np.ndarray:
        """Μετασχηματισμός observation βάσει του τύπου"""
        
        if self.obs_type == 'partial':
            # Κρατάμε μόνο 2 από τα 4 channels (π.χ. τα 2 πιο πρόσφατα frames)
            return state[:2, :, :]
            
        elif self.obs_type == 'noisy':
            # Προσθήκη Gaussian noise
            noise = np.random.normal(0, self.noise_level, state.shape)
            noisy_state = state + noise
            return np.clip(noisy_state, 0, 1)  # Clip στο [0,1] range
            
        elif self.obs_type == 'downsampled':
            # Downsampling κάθε channel
            downsampled = np.zeros((state.shape[0], 42, 42))  # Half size
            for i in range(state.shape[0]):
                downsampled[i] = state[i][::2, ::2]  # Take every 2nd pixel
            return downsampled
            
        else:
            return state
