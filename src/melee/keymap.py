import pygame

MELEE_KEYMAP = {
    # D-pad
    pygame.K_LEFT: "D-LEFT",
    pygame.K_RIGHT: "D-RIGHT",
    pygame.K_DOWN: "D-DOWN",
    pygame.K_UP: "D-UP",

    # Main buttons
    pygame.K_z: "Z",
    pygame.K_r: "R",
    pygame.K_l: "L",
    pygame.K_a: "A",
    pygame.K_b: "B",
    pygame.K_y: "Y",
    pygame.K_x: "X",
    pygame.K_RETURN: "START",
}

# Forbidden combinations that can't be pressed simultaneously
MELEE_FORBIDDEN_COMBINATIONS = [
    {"D-LEFT", "D-RIGHT"},
    {"D-UP", "D-DOWN"},
]

# Analog ranges
MELEE_ANALOG_RANGES = {
    "JOY_X": (-1.0, 1.0),
    "JOY_Y": (-1.0, 1.0),
    "C_X": (-0.9875, 0.9875),
    "C_Y": (-1.0, 1.0),
    "TRIGGER": (0.0, 1.0)
}

# Button list in order matching the neural network output
MELEE_BUTTON_LIST = [
    'D-LEFT', 'D-RIGHT', 'D-DOWN', 'D-UP',
    'Z', 'R', 'L', 'A', 'B', 'Y', 'X', 'START'
]

# Number of discrete positions for analog inputs
MELEE_ANALOG_POSITIONS = 16
MELEE_TRIGGER_POSITIONS = 4