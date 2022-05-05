#! /usr/bin/python3

import sys
import numpy as np


def givens_rotations(a, b, c, d):
    """Calculates the angles needed for a Givens rotation to out put the state with amplitudes a,b,c and d

    Args:
        - a,b,c,d (float): real numbers which represent the amplitude of the relevant basis states (see problem statement). Assume they are normalized.

    Returns:
        - (list(float)): a list of real numbers ranging in the intervals provided in the challenge statement, which represent the angles in the Givens rotations,
        in order, that must be applied.
    """

    # QHACK #
    theta_3 = 2*np.arctan2(-d,a)
    
    if theta_3 > np.pi:
        theta_3 = theta_3 - 2*np.pi
    if theta_3 < -np.pi:
        theta_3 = theta_3 + 2*np.pi
        
        
    theta_2 = 2*np.arctan2(-c,b)
    
    if theta_2 > np.pi:
        theta_2 = theta_2 - 2*np.pi
    if theta_2 < -np.pi:
        theta_2 = theta_2 + 2*np.pi
    
    
    theta_1 = 2*np.arctan2(np.sin(theta_3/2)/np.cos(theta_2/2)*b,d)
    
    if theta_1 > np.pi:
        theta_1 = theta_1 - 2*np.pi
    if theta_1 < -np.pi:
        theta_1 = theta_1 + 2*np.pi
        
    return [theta_1,theta_2,theta_3]
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    theta_1, theta_2, theta_3 = givens_rotations(
        float(inputs[0]), float(inputs[1]), float(inputs[2]), float(inputs[3])
    )
    print(*[theta_1, theta_2, theta_3], sep=",")
