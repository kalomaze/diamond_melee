"""
Credits: some parts are taken and modified from the file `config.py` from https://github.com/TeaPearce/Counter-Strike_Behavioural_Cloning/
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np
import pygame
import torch

from .keymap import (
    MELEE_KEYMAP,
    MELEE_FORBIDDEN_COMBINATIONS,
    MELEE_ANALOG_RANGES,
    MELEE_BUTTON_LIST,
    MELEE_ANALOG_POSITIONS,
    MELEE_TRIGGER_POSITIONS
)

@dataclass
class MeleeAction:
    buttons: List[str]
    joy_x: float
    joy_y: float
    c_x: float
    c_y: float
    trigger: float

    def __post_init__(self) -> None:
        self.buttons = filter_buttons_forbidden(self.buttons)
        self.process_sticks()

    def process_sticks(self) -> None:
        # Clip analog values to valid ranges
        self.joy_x = np.clip(self.joy_x, MELEE_ANALOG_RANGES["JOY_X"][0], MELEE_ANALOG_RANGES["JOY_X"][1])
        self.joy_y = np.clip(self.joy_y, MELEE_ANALOG_RANGES["JOY_Y"][0], MELEE_ANALOG_RANGES["JOY_Y"][1])
        self.c_x = np.clip(self.c_x, MELEE_ANALOG_RANGES["C_X"][0], MELEE_ANALOG_RANGES["C_X"][1])
        self.c_y = np.clip(self.c_y, MELEE_ANALOG_RANGES["C_Y"][0], MELEE_ANALOG_RANGES["C_Y"][1])
        self.trigger = np.clip(self.trigger, MELEE_ANALOG_RANGES["TRIGGER"][0], MELEE_ANALOG_RANGES["TRIGGER"][1])

    @property
    def button_names(self) -> List[str]:
        return [pygame.key.name(key) for key in self.buttons]

def print_melee_action(action: MeleeAction) -> Tuple[str]:
    buttons = " + ".join(action.button_names) if len(action.buttons) > 0 else ""
    main_stick = f"Joy:({action.joy_x:.3f}, {action.joy_y:.3f})"
    c_stick = f"C:({action.c_x:.3f}, {action.c_y:.3f})"
    trigger = f"L/R:{action.trigger:.3f}"
    return buttons, main_stick, c_stick, trigger

def encode_melee_action(melee_action: MeleeAction, device: torch.device) -> torch.Tensor:
    # Initialize one-hot vectors
    buttons_onehot = np.zeros(len(MELEE_BUTTON_LIST))
    joy_x_onehot = np.zeros(MELEE_ANALOG_POSITIONS)
    joy_y_onehot = np.zeros(MELEE_ANALOG_POSITIONS)
    c_x_onehot = np.zeros(MELEE_ANALOG_POSITIONS)
    c_y_onehot = np.zeros(MELEE_ANALOG_POSITIONS)
    trigger_onehot = np.zeros(MELEE_TRIGGER_POSITIONS)

    # Encode buttons
    for button in melee_action.button_names:
        if button in MELEE_BUTTON_LIST:
            buttons_onehot[MELEE_BUTTON_LIST.index(button)] = 1

    # Helper function to encode analog values
    def analog_to_onehot(value: float, n_buckets: int, min_val: float, max_val: float) -> np.ndarray:
        onehot = np.zeros(n_buckets)
        bucket = int((value - min_val) / (max_val - min_val) * (n_buckets - 1))
        bucket = np.clip(bucket, 0, n_buckets - 1)
        onehot[bucket] = 1
        return onehot

    # Encode analog values
    joy_x_onehot = analog_to_onehot(
        melee_action.joy_x,
        MELEE_ANALOG_POSITIONS,
        MELEE_ANALOG_RANGES["JOY_X"][0],
        MELEE_ANALOG_RANGES["JOY_X"][1]
    )

    joy_y_onehot = analog_to_onehot(
        melee_action.joy_y,
        MELEE_ANALOG_POSITIONS,
        MELEE_ANALOG_RANGES["JOY_Y"][0],
        MELEE_ANALOG_RANGES["JOY_Y"][1]
    )

    c_x_onehot = analog_to_onehot(
        melee_action.c_x,
        MELEE_ANALOG_POSITIONS,
        MELEE_ANALOG_RANGES["C_X"][0],
        MELEE_ANALOG_RANGES["C_X"][1]
    )

    c_y_onehot = analog_to_onehot(
        melee_action.c_y,
        MELEE_ANALOG_POSITIONS,
        MELEE_ANALOG_RANGES["C_Y"][0],
        MELEE_ANALOG_RANGES["C_Y"][1]
    )

    trigger_onehot = analog_to_onehot(
        melee_action.trigger,
        MELEE_TRIGGER_POSITIONS,
        MELEE_ANALOG_RANGES["TRIGGER"][0],
        MELEE_ANALOG_RANGES["TRIGGER"][1]
    )

    return torch.tensor(
        np.concatenate((
            buttons_onehot,
            joy_x_onehot,
            joy_y_onehot,
            c_x_onehot,
            c_y_onehot,
            trigger_onehot
        )),
        device=device,
        dtype=torch.float32,
    )

def decode_melee_action(y_preds: torch.Tensor) -> MeleeAction:
    y_preds = y_preds.squeeze()

    # Split prediction vector into components
    n_buttons = len(MELEE_BUTTON_LIST)
    buttons_pred = y_preds[0:n_buttons]
    joy_x_pred = y_preds[n_buttons:n_buttons + MELEE_ANALOG_POSITIONS]
    joy_y_pred = y_preds[n_buttons + MELEE_ANALOG_POSITIONS:n_buttons + 2*MELEE_ANALOG_POSITIONS]
    c_x_pred = y_preds[n_buttons + 2*MELEE_ANALOG_POSITIONS:n_buttons + 3*MELEE_ANALOG_POSITIONS]
    c_y_pred = y_preds[n_buttons + 3*MELEE_ANALOG_POSITIONS:n_buttons + 4*MELEE_ANALOG_POSITIONS]
    trigger_pred = y_preds[n_buttons + 4*MELEE_ANALOG_POSITIONS:]

    # Decode buttons
    buttons_pressed = []
    buttons_onehot = np.round(buttons_pred)
    for i, pressed in enumerate(buttons_onehot):
        if pressed == 1:
            button_name = MELEE_BUTTON_LIST[i]
            button_key = pygame.key.key_code(button_name)
            buttons_pressed.append(button_key)

    # Helper function to decode analog values
    def onehot_to_analog(onehot: np.ndarray, min_val: float, max_val: float) -> float:
        bucket = np.argmax(onehot)
        return min_val + (max_val - min_val) * (bucket / (len(onehot) - 1))

    # Decode analog values
    joy_x = onehot_to_analog(
        joy_x_pred,
        MELEE_ANALOG_RANGES["JOY_X"][0],
        MELEE_ANALOG_RANGES["JOY_X"][1]
    )

    joy_y = onehot_to_analog(
        joy_y_pred,
        MELEE_ANALOG_RANGES["JOY_Y"][0],
        MELEE_ANALOG_RANGES["JOY_Y"][1]
    )

    c_x = onehot_to_analog(
        c_x_pred,
        MELEE_ANALOG_RANGES["C_X"][0],
        MELEE_ANALOG_RANGES["C_X"][1]
    )

    c_y = onehot_to_analog(
        c_y_pred,
        MELEE_ANALOG_RANGES["C_Y"][0],
        MELEE_ANALOG_RANGES["C_Y"][1]
    )

    trigger = onehot_to_analog(
        trigger_pred,
        MELEE_ANALOG_RANGES["TRIGGER"][0],
        MELEE_ANALOG_RANGES["TRIGGER"][1]
    )

    return MeleeAction(buttons_pressed, joy_x, joy_y, c_x, c_y, trigger)

def filter_buttons_forbidden(buttons_pressed: List[str], forbidden_combinations: List[Set[str]] = MELEE_FORBIDDEN_COMBINATIONS) -> List[str]:
    """Filter out illegal button combinations"""
    buttons = set()
    names = set()
    for button in buttons_pressed:
        if button not in MELEE_KEYMAP:
            continue
        name = MELEE_KEYMAP[button]
        buttons.add(button)
        names.add(name)
        for forbidden in forbidden_combinations:
            if forbidden.issubset(names):
                buttons.remove(button)
                names.remove(name)
                break
    return list(filter(lambda button: button in buttons, buttons_pressed))
