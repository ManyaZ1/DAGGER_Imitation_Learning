# utils/expert.py

class ExpertPolicy:
    def __init__(self, env):
        self.env = env

    def act(self, ram):
        # Simple heuristic: if enemy near, jump
        if ram[0x001D] != 0:
            return [0, 0, 1, 0, 0, 0, 0, 0, 0]  # Jump
        return [1, 0, 0, 0, 0, 0, 0, 0, 0]      # Move right
