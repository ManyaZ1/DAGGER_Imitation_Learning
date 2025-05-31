from time import sleep
import cv2
import gym

class MarioRenderer:
    ''' Βοηθητική κλάση για να προβάλλεται το περιβάλλον
        του Gym σε μεγαλύτερη ανάλυση με χρήση OpenCV. '''
    def __init__(self, env: gym.Env, scale: float = 2.) -> None:
        self.env = env
        self.scale = scale

        return

    def render(self) -> None:
        ''' Παίρνει το τρέχον frame από το περιβάλλον και το προβάλλει με
            χρήση OpenCV, μεγενθυμένο σύμφωνα με τον scale factor. '''
        frame = self.env.render(mode = 'rgb_array')

        # Για να μην φαίνονται ξεθωριασμένα τα χρώματα...
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.resize(
            frame,
            (0, 0),
            fx = self.scale,
            fy = self.scale,
            interpolation = cv2.INTER_NEAREST
        )

        cv2.imshow('Mario Agent', frame)
        cv2.waitKey(1)
        sleep(1 / 60) # ~60 FPS

        return
