# -*- coding: utf-8 -*-
"""
@Project :  AI_digital_twin
@Product :  PyCharm
@File    :  algorithm_loader.py
@Time    :  2025/5/12 15:00
@Author  :  Chang_Chao
@Contact :  changchao993@163.com
@Desc    :  support algorithm that simulations may need.
"""

import os
import numpy as np
import shutil
from ansys.aedt.core import Maxwell3d
import logging
from pathlib import Path


def setup_logging():
    """Configure logging for the entire application."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("AEDT_Simulation")
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler(log_dir / "simulation.log", mode='w')
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)s | %(levelname)s | %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '[%(levelname)s] %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    return logger


def delete_simulation_files(max_retries=3):
    """Delete ALL files including .aedt and .aedtresults files.

    This fucntion allows users to manually delete All simulation related files.

    Execution steps:
    1. Remove all name including .aedt project file
    2. Force delete Ansys results directory (including non-empty directories)
    3. if false,
        3.1 create an Maxwell3d instance
        3.2 Close project without saving
        3.3 Close desktop client
        3.4 Repeat step 1 and 2 until all files are deleted.

    """

    project_path = os.getcwd()
    retries = 0
    while retries < max_retries:
        try:
            for root, _, files in os.walk(project_path):
                for file in files:
                    if ".aedt" in file:
                        file_path = os.path.join(root, file)
                        os.remove(file_path)
            for root, dirs, _ in os.walk(project_path):
                for dir_name in dirs:
                    if ".aedtresults" in dir_name:
                        dir_path = os.path.join(root, dir_name)
                        shutil.rmtree(dir_path, ignore_errors=True)
            break
        except:
            maxwell_3d = Maxwell3d(close_on_exit=True,
                                   non_graphical=True,
                                   )
            maxwell_3d.close_project(save=False)
            maxwell_3d.close_desktop()
    return True


def excitation_voltage_to_current_copper(voltage: float = 4,
                                         num_coil: int = 100,
                                         inner_diameter: float = 2.15e-3,
                                         outer_diameter: float = 10e-3,
                                         coil_width: float = 0.125e-3,
                                         coil_height: float = 26e-6,
                                         coil_distance: float = 0.125e-3,
                                         two_coil_distance: float = 4e-3,
                                         frequency: float = 1e6):
    """Calculate the equivalent current per turn of the copper coil.

    This function allows users to calculate the equivalent current per turn of the copper coil. Current is
    determined by the voltage, number of coil, inner diameter, outer diameter, coil width, coil height,
    and frequency. Be aware that the coil material is copper. The unit of the voltage is V, the unit of
    the inner diameter, outer diameter, coil width, coil height is m, and the unit of the frequency is Hz.
    See paper 'Simple Accurate Expressions for Planar Spiral Inductances'.

    Caution:
    ----------
    This is a beta version of the function, it may not be accurate enough for some cases.

    Parameters:
    ----------
    voltage: float,
        The voltage of the excitation. The unit is V. The default is 10.
    num_coil: int,
        The number of coil. The default is 100.
    inner_diameter: float,
        The inner diameter of the coil. The unit is m. The default is 0.005 m.
    outer_diameter: float,
        The outer diameter of the coil. The unit is m. The default is 0.02 m.
    coil_width: float,
        The width of the coil. The unit is m. The default is 0.0001 m.
    coil_height: float,
        The height of the coil. The unit is m. The default is 0.000036 m.
    frequency: float,
        The frequency of the excitation. The unit is Hz. The default is 1 MHz.

    Return:
    -------
    I: float,
        The equivalent current per turn of the copper coil.

    Example:
    --------
    Calculate the equivalent current per turn of the copper coil with 10 V, 100 coils, inner diameter 5 mm,
    outer diameter 20 mm, coil width 0.1 mm, coil height 36 um, and frequency 1 MHz.

    >>> excitation_voltage_to_current_copper(voltage=10, num_coil=100, inner_diameter=0.005, outer_diameter=0.02,
    ...                                      coil_width=0.0001, coil_height=0.000036, frequency=1e6)

    """
    trace_length = num_coil * np.pi * (inner_diameter + num_coil * coil_distance)
    Rc = 1.68e-8 * trace_length / (coil_width * coil_height)
    D = (outer_diameter + inner_diameter) / 2
    fi = (coil_width + coil_distance) / D
    L = ((4 * np.pi * 1e-7 * num_coil ** 2 * D) / 2) * (np.log(2.46 / fi) + 0.2 * fi ** 2)
    X_L = 2 * np.pi * frequency * (L * 1.3)
    Z = np.sqrt(Rc ** 2 + X_L ** 2)
    I = (voltage / Z) * num_coil
    return I, X_L, L, Rc
