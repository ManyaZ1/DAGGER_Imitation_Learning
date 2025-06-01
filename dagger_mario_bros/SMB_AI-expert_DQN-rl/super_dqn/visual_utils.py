from time import sleep
import cv2
import gym
import os

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

class NESControllerOverlay:
    def __init__(self) -> None:
        self.image_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'Nes_controller.png'
        )
        self.button_radius = 29

        # Default button positions (you can override via argument)
        self.button_coords = {
            'A':      (508, 178),
            'B':      (430, 178),
            'right':  (140, 149),
            'left':   (62,  149)
        }

        # Φόρτωσε την εικόνα του NES controller 1 μόνο μία φορά!
        self.base_img = cv2.imread(self.image_path)

        return

    def show(self, pressed_buttons: list) -> None:
        overlay = self.base_img.copy()

        for btn in pressed_buttons:
            if btn in self.button_coords:
                x, y = self.button_coords[btn]
                cv2.circle(
                    overlay,
                    (x, y),
                    self.button_radius,
                    (0, 255, 255),
                    -1
                )

        blended = cv2.addWeighted(overlay, 0.8, self.base_img, 0.5, 0)
        cv2.imshow('NES Controller Overlay', blended)
        cv2.waitKey(1)

        return

if __name__ == '__main__':
    # Testing NESControllerOverlay button placement!
    test = NESControllerOverlay()
    test.show(['A', 'B', 'right', 'left'])
    sleep(20)
