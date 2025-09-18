# -*- coding: utf-8 -*-
"""
@Project :  AI_digital_twin
@Product :  PyCharm
@File    :  simulation_loader.py
@Time    :  2025/2/26 15:00
@Author  :  Chang_Chao
@Contact :  changchao993@163.com
@Desc    :  simulation on digital world
"""

import os
import sys
import numpy as np
from model.algorithm_loader import setup_logging
from ansys.aedt.core import Maxwell3d
from typing import Union, List
import pandas as pd
import logging
import time

logger = setup_logging()


class SimulationInit:
    """Initialize the simulation environment.

    This class allows users to create a new project and design in Maxwell 3D.
    Can create multiple instances of SimulationInit but not recommended.

    Parameters:
    ----------
    project_path: str,
        The path of the project. The default is the current working directory.
    project_name: str,
        The name of the project. The default is 'MyProject'.
    design_name: str,
        The name of the design. The default is 'MyDesign'.
    solver_type: str,
        The type of solver, default is 'Transient',
        can be 'Transient' or 'EddyCurrent'. Equivalent to the
        Time Domain Solver and Frequency Domain Solver in Comsol.
    non_graphical: bool,
        whether to open the Ansys EM GUI, default is False, which means open the GUI.

    Example:
    --------
    Create an instance of Maxwell 3D with the project name 'test_project'
    and design name 'test_design'.

    >>> from model.simulation_loader import SimulationInit
    >>> sim = SimulationInit(project_name='test_project_0', design_name='test_design_0')

    """

    # collect all instances of SimulationInit
    _instances = []

    def __init__(self, project_path: str = os.getcwd(),
                 project_name: str = "MyProject",
                 design_name: str = "MyDesign",
                 solver_type: str = 'Transient',
                 non_graphical: bool = False):
        self.logger = logging.getLogger("AEDT_Simulation.SimulationInit")
        self.logger.info(f"Initializing simulation: {project_name}/{design_name}")
        self.project_path = project_path
        self.project_name = project_name
        self.design_name = design_name
        self.non_graphical = non_graphical
        self.solution_type = solver_type
        self.maxwell_3d = None
        self._coil_names = []
        self._cylinder_names = []
        self._box_names = []
        self._coil_for_assign = []
        self._crack_counter = 0
        self._specimen_name = []
        self._boundary = []
        self._variable_names = [{"x": "0_mm"}, {"y": "0_mm"}, {"z": "0_mm"}]
        self.__class__._instances.append(self)

    def simulation_init(self) -> Maxwell3d:
        """Initialize the simulation environment.

        This function calls the Maxwell 3D class to create a new design instance.

        Returns:
        -------
        maxwell_3d: ansys.aedt.core.Maxwell3d, call an instance of Maxwell 3D.

        Example:
        --------
        Call an instance of Maxwell 3D, the project path is the current working directory,
        the project name is 'test_project', the design name is 'test_design',
        the solver type is 'EddyCurrent'.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_1', design_name='test_design_1')
        >>> m3d = sim.simulation_init()

        """

        if self.solution_type not in {'Transient', 'EddyCurrent'}:
            raise ValueError("The solver type must be 'Transient' or 'EddyCurrent'.")
        self.maxwell_3d = Maxwell3d(
            project=os.path.join(self.project_path, f"{self.project_name}.aedt"),
            design=self.design_name,
            solution_type=self.solution_type,
            non_graphical=self.non_graphical,
            close_on_exit=True,
        )
        self.maxwell_3d.modeler.model_units = "mm"
        self.maxwell_3d["$x"] = "0 mm"
        self.maxwell_3d["$y"] = "0 mm"
        self.maxwell_3d["$z"] = "0 mm"
        return self.maxwell_3d

    def create_project_variable(self,
                                variable_name: str = None,
                                variable_value: float = 0,
                                variable_unit: str = None,
                                **kwargs):
        """Create a project variable in existed project.

        This function allows users to create a project variable in Maxwell 3D.

        Return:
        -------
        True: bool,
        If the project variable create successful, return True.

        Parameters:
        ----------
        variable_name: str,
            The name of the variable. The default is None. If the variable name already exists,
            a ValueError will be raised.
        variable_value: float,
            The value of the variable. The default is None. The default is 0.
        variable_unit: str,
            The unit of the variable. The default is None. The unit should be a string and Can be recognized by Ansys,
            please refer to Ansys Help to see the available units.

        Example:
        --------
        After initialize the project, create two project variable named 'x_move' with value 1.0 and unit 'mm' and
        and "custom_freq" with value 1.0 and unit 'GHz'.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_2', design_name='test_design_2')
        >>> m3d = sim.simulation_init()
        >>> _ = sim.create_project_variable(variable_name='x_move', variable_value=1.0, variable_unit='mm')
        >>> sim.create_project_variable(variable_name='custom_freq', variable_value=1.0, variable_unit='GHz')
        True
        """
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if all(variable_name in var_dict for var_dict in self._variable_names):
            raise ValueError(f"variable name: '{variable_name}' already existed. Please using another variable name.")
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if variable_name is not None and not isinstance(variable_name, str):
            raise ValueError("variable_name must be a string or None")
        if variable_unit is not None and not isinstance(variable_unit, str):
            raise ValueError("variable_unit must be a string or None")
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        self.maxwell_3d["$" + variable_name] = str(variable_value) + variable_unit
        self._variable_names.append({variable_name: str(variable_value) + "_" + variable_unit})
        return True

    def create_rectangle_coil(self,
                              name: str = "MyRecCoil",
                              material: str = "Copper",
                              num_turns: int = 5,
                              step_size: float = 0.25,
                              wire_height: float = 0.035,
                              wire_width: float = 0.125,
                              initial_x_length: float = 1,
                              initial_y_length: float = 1,
                              center: tuple = (0, 0, 0),
                              **kwargs):
        """Create a rectangular coil in existed project.

        This function allows users to create a rectangular coil in Maxwell 3D. The coil is created
        by sweeping a custom rectangle (the cross section of the coil) along a path. Be aware that
        the path is through the center of the rectangle. The coil is assigned a material and moved
        to the desired position. The coil position is the global coordinate system value of the
        center point of the coil.

        Parameters:
        ----------
        name: str,
            The name of the coil. The default is 'MyCoil'. Each name of coils must be unique.
            If the name already exists, a ValueError will be raised.
        material: str,
            The material of the coil. The default is 'Copper'. Please refer to the material library
            in Maxwell 3D for the available materials.
        num_turns: int,
            The number of turns of the coil. The default is 5.
        step_size: float,
            The step size of the coil, which means the distance between two adjacent center lines of turns
            of the coil. The default is 0.25 mm. Be aware that the step size should greater than
            the wire width.
        wire_height: float,
            The height of each turn of coil. The default is 0.035 mm.
        wire_width: float,
            The width of each turn of coil. The default is 0.125 mm. Be aware that the wire width
            should smaller than the step size.
        initial_x_length: float,
            The initial length of the coil in x direction, which means the size of the innermost coil.
            The default is 1 mm.
        initial_x_length: float,
            The initial length of the coil in y direction, which means the size of the innermost coil.
            The default is 1 mm.
        center: tuple,
            The position of the coil. Determined by the center of the coil.
            The default is (0, 0, 0). The unit is mm. Which means the center of the coil's position
            are all 0 mm in x, y, z direction, measured in global coordinate system.

        Returns:
        -------
        coil_for_assign: cross section of the coil for assign excitation.

        Example:
        --------
        After initialize the project, create a rectangular coil named 'RecCoil' with
        10 turns and the center position is (1, 0, 1). The coil is assigned a material
        'Aluminum'.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_3', design_name='test_design_3')
        >>> m3d = sim.simulation_init()
        >>> coil_for_assign = sim.create_rectangle_coil(name='RecCoil', material='Aluminum',
        ...                                              num_turns=10, center=(1, 0, 1))

        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if "_" in name:
            raise ValueError("The name of the coil should not contain '_'.")
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if name in self._coil_names:
            raise ValueError(f"coil name: '{name}' already existed. Please using another name.")
        if step_size <= wire_width:
            raise ValueError("The step size must be greater than the wire width.")
        self._coil_names.append(name)
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        x, y = [0], [0]
        x_length = initial_x_length
        y_length = initial_y_length
        for i in range(num_turns):
            x.append(x[-1] + x_length)
            y.append(y[-1])
            x_length += step_size
            x.append(x[-1])
            y.append(y[-1] + y_length)
            y_length += step_size
            x.append(x[-1] - x_length)
            y.append(y[-1])
            x_length += step_size
            x.append(x[-1])
            y.append(y[-1] - y_length)
            y_length += step_size
        x.append(x[0] + wire_height)
        y.append(y[-1])
        z = np.zeros_like(x)
        path = list(zip(x, y, z))
        self.maxwell_3d.modeler.create_polyline(path, cover_surface=False,
                                                name=name + "_coil_path")
        self.maxwell_3d.modeler.create_rectangle(
            orientation="YZ",
            origin=[x[0] + 0.5, y[0] - wire_width / 2, z[0] - wire_height / 2],
            sizes=[wire_width, wire_height],
            name=name
        )
        coil_part = self.maxwell_3d.modeler.sweep_along_path(name,
                                                             name + "_coil_path")
        loop = [
            (x[0], y[0], z[0]),
            (x[0], y[0], z[0] + wire_height * 3),
            (x[-1], y[-1], z[-1] + wire_height * 3),
            (x[-1], y[-1], z[-1] - wire_height)
        ]
        self.maxwell_3d.modeler.create_polyline(loop, cover_surface=False,
                                                name=name + "_loop_path")
        loop_section = self.maxwell_3d.modeler.create_rectangle(
            orientation="XY",
            origin=[x[0] - wire_height / 2, y[0] - wire_width / 2, -wire_height / 2],
            sizes=[wire_height, wire_width],
            name=name + "_loop"
        )
        loop_part = self.maxwell_3d.modeler.sweep_along_path(name + "_loop",
                                                             name + "_loop_path")
        coil = self.maxwell_3d.modeler.unite([coil_part, loop_part])
        self.maxwell_3d.assign_material(coil, material)
        self.maxwell_3d.modeler.move(
            name,
            [center[0], center[1], center[2]])
        self.maxwell_3d.modeler.create_coordinate_system(
            name=name + "_for_section",
            origin=(center[0], center[1], center[2] + 0.5 * wire_height),
        )
        self.maxwell_3d.modeler.set_working_coordinate_system(name + "_for_section")
        self.maxwell_3d.modeler.section(coil, "XY")
        all_section_objects = (self.maxwell_3d.modeler.get_objects_w_string(name) and
                               self.maxwell_3d.modeler.get_objects_w_string("Section"))
        all_section_objects = [
            item for item in all_section_objects
            if (name in item) and ("Section" in item)
        ]
        for obj in all_section_objects:
            self.maxwell_3d.modeler.separate_bodies(obj)
        all_separate_objects = self.maxwell_3d.modeler.get_objects_w_string("Separate")
        for obj in all_separate_objects:
            self.maxwell_3d.modeler.delete(obj)
        coil_for_assign = [obj for obj in self.maxwell_3d.modeler.get_objects_w_string("") if
                           name in obj and "Section" in obj]
        self._coil_for_assign.append(coil_for_assign)
        return coil_for_assign

    def create_spiral_coil(self,
                           name: str = "MySpirCoil",
                           material: str = "Copper",
                           num_turns: int = 5,
                           wire_height: float = 0.035,
                           wire_width: float = 0.125,
                           spacing: float = 0.25,
                           coil_inner_radius: float = 1,
                           center: tuple = (0, 0, 0),
                           **kwargs):
        """Create a spiral coil in existed project.

        This function allows users to create a spiral coil in Maxwell 3D. The coil is created
        by sweeping a custom rectangle (the cross section of the coil) along a path. Be aware that
        the path is through the center of the rectangle. The coil is assigned a material and moved
        to the desired position. The coil position is the global coordinate system value of the center
        point of the coil.

        Parameters:
        ----------
        name: str,
            The name of the coil. The default is 'MyCoil'. Each name of coils must be unique.
            If the name already exists, a ValueError will be raised.
        material: str,
            The material of the coil. The default is 'Copper'. Please refer to the material library
            in Maxwell 3D for the available materials.
        num_turns: int,
            The number of turns of the coil. The default is 5.
        wire_height: float,
            The height of each turn of coil. The default is 0.035 mm.
        wire_width: float,
            The width of each turn of coil. The default is 0.125 mm.
        spacing: float,
            The distance between two adjacent turns of the coil of center line. The default is 0.25 mm.
        coil_inner_radius: float,
            The radius of the innermost coil. The default is 1 mm.
        center: tuple,
            The position of the coil. Determined by the center of the coil.
            The default is (0, 0, 0). The unit is mm. Which means the center of the coil's position
            are all 0 mm in x, y, z direction in global coordinate system.

        Returns:
        -------
        coil_for_assign: cross-section of the coil for assign excitation.

        Example:
        --------
        After initialize the project, create a spiral coil named 'SpirCoil' with
        10 turns, and the center position is (1, 0, 1). The coil is assigned a
        material 'Aluminum'.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_4', design_name='test_design_4')
        >>> m3d = sim.simulation_init()
        >>> coil_for_assign = sim.create_spiral_coil(name='SpirCoil', material='Aluminum',
        ...                                              num_turns=10, center=(1, 0, 1))

        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if "_" in name:
            raise ValueError("The name of the coil should not contain '_'.")
        if spacing <= wire_width:
            raise ValueError("The spacing must be greater than the wire width.")
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if name in self._coil_names:
            raise ValueError(f"coil name: '{name}' already existed. Please using another name.")
        self._coil_names.append(name)
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        theta = np.linspace(0, 2 * np.pi * num_turns, 50 * num_turns)
        x = (coil_inner_radius + spacing * theta / (2 * np.pi)) * np.cos(theta)
        y = (coil_inner_radius + spacing * theta / (2 * np.pi)) * np.sin(theta)
        z = np.zeros_like(x)
        path = list(zip(x, y, z))
        self.maxwell_3d.modeler.create_polyline(path, cover_surface=False,
                                                name=name + "_coil_path")
        self.maxwell_3d.modeler.create_rectangle(
            orientation="XZ",
            origin=[x[0] - (wire_width / 2), 0, z[0] - (wire_height / 2)],
            sizes=[wire_height, wire_width],
            name=name
        )
        coil_part = self.maxwell_3d.modeler.sweep_along_path(name,
                                                             name + "_coil_path")
        loop = [
            (x[0], y[0], z[0]),
            (x[0], y[0], z[0] + wire_height * 3),
            # (x[-1] + (wire_width / 2), y[-1], z[0] + wire_height * 2),
            # (x[-1] + (wire_width / 2), y[-1], z[0] - wire_height)
            (x[-1], y[-1], z[0] + wire_height * 3),
            (x[-1], y[-1], z[-1] - wire_height),
        ]
        self.maxwell_3d.modeler.create_polyline(loop, cover_surface=False,
                                                name=name + "_loop_path")
        loop_section = self.maxwell_3d.modeler.create_circle(
            orientation="XY",
            origin=[x[0], y[0], -wire_height / 2],
            radius=min(wire_width, wire_height),
            name=name + "_loop"
        )
        loop_part = self.maxwell_3d.modeler.sweep_along_path(name + "_loop",
                                                             name + "_loop_path")
        coil = self.maxwell_3d.modeler.unite([coil_part, loop_part])
        self.maxwell_3d.assign_material(coil, material)
        self.maxwell_3d.modeler.move(
            name,
            [center[0], center[1], center[2]])
        self.maxwell_3d.modeler.create_coordinate_system(
            name=name + "_for_section",
            origin=(center[0], center[1], center[2] + wire_height),
        )
        self.maxwell_3d.modeler.set_working_coordinate_system(name + "_for_section")
        self.maxwell_3d.modeler.section(coil, "XY")
        all_section_objects = (self.maxwell_3d.modeler.get_objects_w_string(name) and
                               self.maxwell_3d.modeler.get_objects_w_string("Section"))
        all_section_objects = [
            item for item in all_section_objects
            if (name in item) and ("Section" in item)
        ]
        for obj in all_section_objects:
            self.maxwell_3d.modeler.separate_bodies(obj)
        all_separate_objects = self.maxwell_3d.modeler.get_objects_w_string("Separate")
        for obj in all_separate_objects:
            self.maxwell_3d.modeler.delete(obj)
        coil_for_assign = [obj for obj in self.maxwell_3d.modeler.get_objects_w_string("") if
                           name in obj and "Section" in obj]
        self._coil_for_assign.append(coil_for_assign)
        return coil_for_assign

    def create_helmholtz_coils(self,
                               name: str = "MyHelCoil",
                               material: str = "Copper",
                               inner_diameter: float = 5,
                               outer_diameter: float = 20,
                               height: float = 5,
                               center: tuple = (0, 0, 0),
                               **kwargs
                               ):
        """Create a Helmholtz coil in existed project.

        This function allows users to create a Helmholtz coil in Maxwell 3D. The coil is created
        by creating two cylinders and subtracting the inner cylinder from the outer cylinder.

        Parameters:
        ----------
        name: str,
            The name of the coil. The default is 'MyHelCoil'. Each name of coils must be unique.
            If the name already exists, a ValueError will be raised.
        material: str,
            The material of the coil. The default is 'Copper'. Please refer to the material library
            in Maxwell 3D for the available materials.
        inner_diameter: float,
            The inner diameter of the coil. The default is 5 mm.
        outer_diameter: float,
            The outer diameter of the coil. The default is 20 mm.
        height: float,
            The height of the coil. The default is 5 mm.
        center: tuple,
            The position of the coil. Determined by the center of the coil.
            The default is (0, 0, 0). The unit is mm. Which means the center of the coil's position
            are all 0 mm in x, y, z direction in global coordinate system.

        Returns:
        -------
        coil_for_assign: cross-section of the coil for assign excitation.

        Example:
        --------
        After initialize the project, create a Helmholtz coil named 'HelCoil' with
        inner diameter 5 mm, outer diameter 20 mm, and the center position is (1, 0, 1).
        The coil is assigned a material 'Aluminum'.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_5', design_name='test_design_5')
        >>> m3d = sim.simulation_init()
        >>> coil_for_assign = sim.create_helmholtz_coils(name='HelCoil', material='Aluminum',
        ...                                              inner_diameter=5, outer_diameter=20,
        ...                                              center=(1, 0, 1))

        """
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if "_" in name:
            raise ValueError("The name of the coil should not contain '_'.")
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if name in self._coil_names:
            raise ValueError(f"coil name: '{name}' already existed. Please using another name.")
        self._coil_names.append(name)
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        self.maxwell_3d.modeler.create_cylinder(orientation='Z',
                                                origin=list(center),
                                                radius=inner_diameter / 2,
                                                height=height,
                                                name=name + '_inner',
                                                material="Air")
        self.maxwell_3d.modeler.create_cylinder(orientation='Z',
                                                origin=list(center),
                                                radius=outer_diameter / 2,
                                                height=height,
                                                name=name,
                                                material=material)
        self.maxwell_3d.modeler.subtract(name, name + '_inner', keep_originals=False)
        self.maxwell_3d.modeler.create_coordinate_system(
            name=name + "_for_section",
            origin=(center[0], center[1], center[2]),
        )
        self.maxwell_3d.modeler.set_working_coordinate_system(name + "_for_section")
        self.maxwell_3d.modeler.section(name, "XZ")
        all_section_objects = (self.maxwell_3d.modeler.get_objects_w_string(name) and
                               self.maxwell_3d.modeler.get_objects_w_string("Section"))
        all_section_objects = [
            item for item in all_section_objects
            if (name in item) and ("Section" in item)
        ]
        for obj in all_section_objects:
            self.maxwell_3d.modeler.separate_bodies(obj)
        all_separate_objects = self.maxwell_3d.modeler.get_objects_w_string("Separate")
        for obj in all_separate_objects:
            self.maxwell_3d.modeler.delete(obj)
        coil_for_assign = [obj for obj in self.maxwell_3d.modeler.get_objects_w_string("") if
                           name in obj and "Section" in obj]
        self._coil_for_assign.append(coil_for_assign)
        return coil_for_assign

    def create_circular_litz_coils(self,
                                   name: str = "MyHelCoil",
                                   material: str = "Copper",
                                   inner_diameter: float = 5,
                                   outer_diameter: float = 20,
                                   height: float = 5,
                                   type: str = "Rectangular",
                                   wire_diameter: float = 0.125,
                                   wire_width: float = 0.125,
                                   wire_height: float = 0.035,
                                   wire_num: int = 100,
                                   center: tuple = (0, 0, 0),
                                   **kwargs
                                   ):
        """Create a litz wire coil in existed project.

        This function allows users to create a litz wire coil in Maxwell 3D. The coil is created
        by creating two cylinders and subtracting the inner cylinder from the outer cylinder, the
        assign the materials with litz wire property.

        Parameters:
        ----------
        name: str,
            The name of the coil. The default is 'MyHelCoil'. Each name of coils must be unique.
            If the name already exists, a ValueError will be raised.
        material: str,
            The material of the coil. The default is 'Copper'. Please refer to the material library
            in Maxwell 3D for the available materials.
        inner_diameter: float,
            The inner diameter of the coil. The default is 5 mm.
        outer_diameter: float,
            The outer diameter of the coil. The default is 20 mm.
        height: float,
            The height of the coil. The default is 5 mm.
        type: str,
            The type of the each litz wire. The default is 'Rectangular'. Can be 'Rectangular' or 'Round'.
            Please refer to the Ansys help document for more details.
        wire_diameter: float,
            The wire diameter of each number of litz wire. The default is 0.125 mm. Ignored if the type is
            'Rectangular'. Please see more details in Ansys help document.
        wire_width: float,
            The wire width of each number of litz wire. The default is 0.125 mm. Ignored if the type is 'Round'.
            Please see more details in Ansys help document.
        wire_height: float,
            The wire height of each number of litz wire. The default is 0.035 mm. Ignored if the type is 'Round'.
            Please see more details in Ansys help document.
        center: tuple,
            The position of the coil. Determined by the center of the coil.
            The default is (0, 0, 0). The unit is mm. Which means the center of the coil's position
            are all 0 mm in x, y, z direction in global coordinate system.

        Returns:
        -------
        coil_for_assign: cross-section of the coil for assign excitation.

        Example:
        --------
        After initialize the project, create a cicular litz coil named 'HelCoil' with
        inner diameter 5 mm, outer diameter 20 mm, and the center position is (1, 0, 1).
        The coil is assigned a material 'Copper'.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_6', design_name='test_design_6')
        >>> m3d = sim.simulation_init()
        >>> coil_for_assign = sim.create_circular_litz_coils(name='LitzCoil', material='Aluminum',
        ...                                              inner_diameter=5, outer_diameter=20,
        ...                                              center=(1, 0, 1))

        """
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if "_" in name:
            raise ValueError("The name of the coil should not contain '_'.")
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if name in self._coil_names:
            raise ValueError(f"coil name: '{name}' already existed. Please using another name.")
        self._coil_names.append(name)
        WireWidth = str(wire_width) + "mm"
        WireHeight = str(wire_height) + "mm"
        WireDiameter = str(wire_diameter) + "mm"
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        self.maxwell_3d.modeler.create_coordinate_system(
            name=name + "_for_section",
            origin=(center[0], center[1], center[2]),
        )
        self.maxwell_3d.modeler.set_working_coordinate_system(name + "_for_section")
        material_for_modify = self.maxwell_3d.materials.exists_material(material)
        if not material_for_modify:
            raise ValueError(f"material: '{material}' not in the Ansys library, please recheck the material.")
        else:
            new_material_name = material + "_" + name
            self.maxwell_3d.materials.duplicate_material(material, new_material_name)
            litz = self.maxwell_3d.materials[new_material_name]
            litz.coordinate_system = "Cylindrical"
            litz.stacking_type = "Litz Wire"
            litz.wire_type = type
            if type == "Rectangular":
                litz.strand_number = wire_num
                litz.wire_width = WireWidth
                litz.wire_thickness = WireHeight
                litz.wire_thickness_direction = "V(3)"
                litz.wire_width_direction = "V(2)"
            elif type == "Round":
                litz.strand_number = wire_num
                litz.wire_diameter = WireDiameter
            else:
                raise ValueError(f"Invalid type: '{type}'. The type should be 'Rectangular' or 'Round'.")
        self.maxwell_3d.modeler.create_cylinder(orientation='Z',
                                                origin=[0, 0, 0],
                                                radius=inner_diameter / 2,
                                                height=height,
                                                name=name + '_inner',
                                                material="Air")
        self.maxwell_3d.modeler.create_cylinder(orientation='Z',
                                                origin=[0, 0, 0],
                                                radius=outer_diameter / 2,
                                                height=height,
                                                name=name,
                                                material="Air")
        self.maxwell_3d.modeler.subtract(name, name + '_inner', keep_originals=False)
        self.maxwell_3d.assign_material(name, new_material_name)
        self.maxwell_3d.modeler.section(name, "XZ")
        all_section_objects_0 = (self.maxwell_3d.modeler.get_objects_w_string(name) and
                                 self.maxwell_3d.modeler.get_objects_w_string("Section"))
        all_section_objects = [
            item for item in all_section_objects_0
            if (name in item) and ("Section" in item)
        ]
        for obj in all_section_objects:
            self.maxwell_3d.modeler.separate_bodies(obj)
        all_separate_objects = self.maxwell_3d.modeler.get_objects_w_string("Separate")
        for obj in all_separate_objects:
            self.maxwell_3d.modeler.delete(obj)
        coil_for_assign = [obj for obj in self.maxwell_3d.modeler.get_objects_w_string("") if
                           name in obj and "Section" in obj]
        self._coil_for_assign.append(coil_for_assign)
        return coil_for_assign

    def create_cylinder(self,
                        out_diameter: float = 10,
                        inner_diameter: float = 5,
                        height: float = 5,
                        name: str = "MyCylinder",
                        material: str = "Copper",
                        center: tuple = (0, 0, 0),
                        **kwargs):
        """Create a cylinder in existed project.
        """
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if name in self._cylinder_names:
            raise ValueError(f"cylinder name: '{name}' already existed. Please using another name.")
        self._cylinder_names.append(name)
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        self.maxwell_3d.modeler.create_cylinder(orientation='Z', origin=list(center), radius=out_diameter / 2,
                                                name=name, height=height, material=material)
        self.maxwell_3d.modeler.create_cylinder(orientation='Z', origin=list(center), radius=inner_diameter / 2,
                                                name=name + "_inner", height=height, material="Air")
        self.maxwell_3d.modeler.subtract(name, name + "_inner", keep_originals=False)

    def create_box(self,
                   x_length: float = 5,
                   y_length: float = 5,
                   z_length: float = 5,
                   name: str = "MyRectangle",
                   center: tuple = (0, 0, 0),
                   material: str = "Air",
                   **kwargs):
        """Create a box in existed project. upper face is XY plane."""
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if name in self._box_names:
            raise ValueError(f"rectangle name: '{name}' already existed. Please using another name.")
        self._box_names.append(name)
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        origin = (center[0] - x_length / 2, center[1] - y_length / 2, center[2])
        self.maxwell_3d.modeler.create_box(orientation='XY', origin=origin,
                                           sizes=[x_length, y_length, z_length],
                                           name=name, material=material)

    def transient_type_assign_excitation(self,
                                         coil_name: str,
                                         excitation_type: str = "Voltage",
                                         conductors_number: int = 100,
                                         resistance: float = 0.001,
                                         excitation_parameters: str = "10*sin(2*pi*1e6*Time+0)",
                                         solid: bool = False,
                                         **kwargs):
        """Assign voltage or current excitation to the coils.

        This function allows users to assign custom voltage or current excitation to the coils.
        The excitation parameters are determined by the user. Be aware that the solver type
        should be 'Transient'.

        Parameters:
        ----------
        coil_name: str,
            The name of the coil. The coil name should already existed.
        excitation_type: str,
            The type of excitation. The default is 'Voltage'. The excitation type can be 'Voltage'
            or 'Current'.
        conductors_number: int,
            The number of conductors in the coil. Which means the number of conductors through the
            single cross-section of the coil. The default is 100.
        resistance: float,
            The resistance of the winding (not the 3D model part but the excitation winding). Which
            means the inner resistance of the excitation winding. The default is 0.001 ohm.
            Please refer to the Ansys Maxwell Help document.
        excitation_parameters: str,
            The excitation parameters. The default is "10*sin(2*pi*1e6*Time+0)". Which means the
            voltage amplitude is 10 V, the frequency is 1 MHz, and the phase is 0. Be aware the Unit
            that should pre-defined in the Ansys EM software. Please refer to the Ansys Maxwell Help
            document.
        solid: bool,
            Whether the excitation winding is solid or strand. The default is False. Please refer to
            the Ansys Maxwell Help document for more details.

        Return:
        -------
        True: bool,
            If the excitation assignment is successful, return True.

        Example:
        --------
        After initialize the project, create a rectangular coil named 'RecCoil' with
        10 turns and the center position is (1, 0, 1). The coil is assigned a material
        'Aluminum'. Assign a voltage excitation winding to the coil with resistance 50 ohm and
        amplititude of 10*sin(2*pi*1e6*Time+0) V.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_7', design_name='test_design_7',
        ...                       solver_type='Transient')
        >>> m3d = sim.simulation_init()
        >>> coil_for_assign = sim.create_rectangle_coil(name='RecCoil', material='Aluminum',
        ...                                              num_turns=10, center=(1, 0, 1))
        >>> sim.transient_type_assign_excitation(coil_name='RecCoil', conductors_number=100,
        ...                                      resistance=50, excitation_parameters="10*sin(2*pi*1e6*Time+0)")
        True

        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if not self.solution_type == 'Transient':
            raise ValueError(f"Solver type should be 'Transient', but got '{self.solution_type}'.")
        target_objects = [
            obj[0] for obj in self._coil_for_assign
            if obj[0].split('_', 1)[0] == coil_name
        ]
        if not target_objects:
            raise ValueError(f"Coil '{coil_name}' not found. Please recheck the coil name")
        self.maxwell_3d.assign_coil(target_objects,
                                    conductors_number=conductors_number,
                                    name=coil_name + "_for_coil")
        if excitation_type == "Voltage":
            self.maxwell_3d.assign_winding(
                winding_type="Voltage",
                is_solid=solid,
                resistance=resistance,
                voltage=excitation_parameters + "V",
                name=coil_name + "_for_winding"
            )
        elif excitation_type == "Current":
            self.maxwell_3d.assign_winding(
                winding_type="Current",
                is_solid=solid,
                current=excitation_parameters + "A",
                name=coil_name + "_for_winding"
            )
        self.maxwell_3d.add_winding_coils(coil_name + "_for_winding",
                                          coil_name + "_for_coil")
        return True

    def ec_type_assign_excitation(self,
                                  coil_name: str,
                                  excitation_type: str = "Voltage",
                                  conductors_number: int = 100,
                                  resistance: float = 0.001,
                                  amplitude: float or str = 10,
                                  phase: float or str = 0.0,
                                  solid: bool = False,
                                  **kwargs):
        """Assign voltage or current excitation to the coils.

        This function allows users to assign custom voltage or current excitation to the coils.
        Be aware that the solver type should be 'EddyCurrent', the excitation frequency
        must be the same and determined in the ec_setup function if users create
        multiple coils at once. Be aware the Unit that should pre-defined in the Ansys EM
        software. Please refer to the Ansys Maxwell Help document.

        Parameters:
        ----------
        coil_name: str,
            The name of the coil. The coil name should already existed.
        excitation_type: str,
            The type of excitation. The default is 'Voltage'. The excitation type can be 'Voltage'
            or 'Current'.
        conductors_number: int,
            The number of conductors in the coil. Which means the number of conductors through the
            single cross-section of the coil. The default is 100.
        resistance: float,
            The resistance of the winding (not the 3D model part but the excitation winding). Which
            means the inner resistance of the excitation winding. The default is 0.001 ohm.
            Please refer to the Ansys Maxwell Help document.
        amplitude: float,
            The excitation amplitude of voltage (V) or current (A). The default is 10.
            Be aware the Unit that should pre-defined in the Ansys EM software. Please refer to
            the Ansys Maxwell Help document.
        phase: float,
            The initial phase of the excitation. The default is 0 deg. Please refer to the Ansys
            Maxwell Help document.
        solid: bool,
            Whether the excitation winding is solid or strand. The default is False. Please refer to
            the Ansys Maxwell Help document for more details.

        Return:
        -------
        True: bool,
            If the excitation assignment is successful, return True.

        Example:
        --------
        After initialize the project, create a spiral coil named 'SpirCoil' with
        10 turns and the center position is (1, 0, 1). The coil is assigned a material
        'Copper'. Assign a voltage excitation widing to the coil with resistance 50 ohm
        and voltage 10 V.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_8', design_name='test_design_8',
        ...                       solver_type='EddyCurrent')
        >>> m3d = sim.simulation_init()
        >>> coil_for_assign = sim.create_spiral_coil(name='SpirCoil', material='Copper',
        ...                                              num_turns=10, center=(1, 0, 1))
        >>> sim.ec_type_assign_excitation(coil_name='SpirCoil', conductors_number=100,
        ...                                      resistance=50, amplitude=10, phase=0)
        True
        """
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if not self.solution_type == 'EddyCurrent':
            raise ValueError(f"Solver type should be 'EddyCurrent', but got '{self.solution_type}'.")
        target_objects = [
            obj[0] for obj in self._coil_for_assign
            if obj[0].split('_', 1)[0] == coil_name
        ]
        if type(amplitude) == str:
            if not any(amplitude in var_dict for var_dict in self._variable_names):
                raise ValueError(f"variable '{amplitude}' not found. Please recheck the variable name")
        if type(phase) == str:
            if not any(phase in var_dict for var_dict in self._variable_names):
                raise ValueError(f"variable '{phase}' not found. Please recheck the variable name")
        if not target_objects:
            raise ValueError(f"Coil '{coil_name}' not found. Please recheck the coil name")
        if excitation_type == "Voltage" and type(amplitude) == float:
            amplitude_str = f"{amplitude} V"
        elif excitation_type == "Voltage" and type(amplitude) == str:
            amplitude_str = "$" + amplitude
        elif excitation_type == "Current" and type(amplitude) == float:
            amplitude_str = f"{amplitude} A"
        elif excitation_type == "Current" and type(amplitude) == str:
            amplitude_str = "$" + amplitude
        if type(phase) == float:
            phase_str = f"{phase} deg"
        elif type(phase) == str:
            phase_str = "$" + phase
        self.maxwell_3d.assign_coil(target_objects,
                                    conductors_number=conductors_number,
                                    name=coil_name + "_for_coil")
        if excitation_type == "Voltage":
            self.maxwell_3d.assign_winding(
                winding_type="Voltage",
                is_solid=solid,
                resistance=resistance,
                voltage=amplitude_str,
                phase=phase_str,
                name=coil_name + "_for_winding"
            )
        elif excitation_type == "Current":
            self.maxwell_3d.assign_winding(
                winding_type="Current",
                is_solid=solid,
                current=amplitude_str,
                phase=phase_str,
                name=coil_name + "_for_winding"
            )
        self.maxwell_3d.add_winding_coils(coil_name + "_for_winding",
                                          coil_name + "_for_coil")
        return True

    def create_specimen(self,
                        specimen_name: str = "MySpecimen",
                        material: str = "Aluminum",
                        length: float = 20,
                        width: float = 10,
                        height: float = 5,
                        **kwargs):
        """Create a specimen with defects.

        This function allow users to create a rectangular specimen in Maxwell 3D. Be aware that the
        center point of the specimen on upper face will be placed in the origin of the global coordinate.

        Parameters:
        ----------
        specimen_name: str,
            The name of the specimen. The default is 'MySpecimen'.
        material: str,
            The material of the specimen. The default is 'Aluminum'. Please refer to
            the material library in Maxwell 3D for the available materials.
        length: float,
            The length of the specimen in x direction. The default is 20 mm.
        width: float,
            The width of the specimen in y direction. The default is 10 mm.
        height: float,
            The height of the specimen in z direction. The default is 5 mm.

        Return:
        -------
        True: bool,
            If the specimen creation is successful, return True.

        Example:
        --------
        After initialize the project, create a specimen named 'MySpecimen' with material 'Aluminum',
        length 20 mm, width 10 mm, and height 5 mm.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_9', design_name='test_design_9')
        >>> m3d = sim.simulation_init()
        >>> sim.create_specimen(specimen_name='MySpecimen', material='Aluminum', length=20, width=10, height=5)
        True
        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if specimen_name in self._specimen_name:
            raise ValueError(f"specimen name: '{specimen_name}' already existed. Please using another name.")
        self._specimen_name.append(specimen_name)
        self.maxwell_3d.modeler.set_working_coordinate_system("Global")
        self.maxwell_3d.modeler.create_box(
            [-length / 2, -width / 2, -height],
            [length, width, height],
            name=specimen_name
        )
        self.maxwell_3d.assign_material(specimen_name, material)
        return True

    def specimen_with_crack(self,
                            specimen_name: str = "MySpecimen",
                            crack_length: float = 0.5,
                            crack_width: float = 0.5,
                            crack_height: float = 0.5,
                            center: tuple = (0, 0, 0),
                            **kwargs):
        """Create a specimen with a crack.

        This function allows users to create a rectangular crack in the specimen. The crack is created
        by subtracting a box from the specimen. Be aware that the center point is determined by the
        crack on the upper face's centre point.

        Parameters:
        ----------
        specimen_name: str,
            The name of the specimen. The default is 'MySpecimen'. Be aware that the specimen name
            must existed in the project.
        crack_length: float,
            The length of the crack in x direction. The default is 0.5 mm.
        crack_width: float,
            The width of the crack in y direction. The default is 0.5 mm.
        crack_height: float,
            The height of the crack in z direction. The default is 0.5 mm.
        center: tuple,
            The position of the crack determined by the center point of the defect upper face.
            The default is (0, 0, 0). The unit is mm. Which means the center of the crack's position
            are all 0 mm in x, y, z direction.

        Return:
        -------
        True: bool,
            If the crack creation is successful, return True.

        Example:
        --------
        After initialize the project and create a specimen, further create a defect in the specimen.
        The defect is a crack with length 0.5 mm, width 0.5 mm, and height 0.5 mm. The center of the
        crack is placed at the origin of the global coordinate.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_a', design_name='test_design_a')
        >>> m3d = sim.simulation_init()
        >>> _ = sim.create_specimen()
        >>> sim.specimen_with_crack(specimen_name='MySpecimen', crack_length=0.5, crack_width=0.5, crack_height=0.5)
        True

        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        crack_center = [center[0] - crack_length / 2,
                        center[1] - crack_width / 2,
                        center[2] - crack_height]
        if self.maxwell_3d is None:
            raise ValueError("Maxwell 3D instance not initialized. Call simulation_init() first.")
        if specimen_name not in self._specimen_name:
            raise ValueError(f"specimen: '{specimen_name}' not exist, choose existed one.")
        crack_name = f"{specimen_name}_crack_{self._crack_counter + 1}"
        self.maxwell_3d.modeler.create_box(crack_center, [crack_length, crack_width, crack_height],
                                           name=crack_name)
        self.maxwell_3d.modeler.subtract([specimen_name], [crack_name], keep_originals=False)
        self._crack_counter += 1
        return True

    def length_mesh(self,
                    object_name: str or list,
                    inside: bool = False,
                    mesh_size: float or list = 0.1,
                    **kwargs):
        """Set up the mesh size of the object.

        This function allows users to assign the mesh size of the object in Maxwell 3D. The mesh size
        is determined by the user. Be aware that the object name should be the existed object in the
        project.

        Parameters:
        ----------
        object_name: str or list,
            The name of the object need assign mesh size. The object name should already existed.
        inside: bool,
            Whether the mesh type is inside or on selection. The default is False.
        mesh_size: float or list,
            The maximum mesh size of the object. The default is 0.1 mm. if the object name and mesh size
            are both list, these two parameters should at the same length. If object name is list and
            mesh size is float, the mesh size will be assigned to all the object name in the list.

        Return:
        -------
        True: bool,
        If the mesh assignment is successful, return True.

        Example:
        --------
        After initialize the project and create a specimen, assign the mesh size of the specimen to 0.1 mm.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_b', design_name='test_design_b')
        >>> m3d = sim.simulation_init()
        >>> _ = sim.create_specimen()
        >>> sim.length_mesh(object_name='MySpecimen', mesh_size=0.1)
        True
        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if type(object_name) == str and (type(mesh_size) == float or type(mesh_size) == int):
            if object_name not in self._coil_names and object_name not in self._specimen_name:
                raise ValueError(f"Object name: '{object_name}' not found. Please recheck the object name.")
            meshsize = f'{mesh_size} mm'
            self.maxwell_3d.mesh.assign_length_mesh(
                assignment=object_name,
                inside_selection=inside,
                maximum_length=meshsize,
                maximum_elements=None,
            )
        elif type(object_name) == list and type(mesh_size) == list:
            for obj in object_name:
                if (obj not in self._coil_names) and (obj not in self._specimen_name):
                    raise ValueError(f"Object name: '{obj}' not found. Please recheck the object name.")
            if len(object_name) != len(mesh_size):
                raise ValueError(f"Please recheck all the object name and its coresponding mesh size")
            else:
                for obj, size in zip(object_name, mesh_size):
                    mesh_str = f"{size} mm"
                    self.maxwell_3d.mesh.assign_length_mesh(
                        assignment=obj,
                        inside_selection=inside,
                        maximum_length=mesh_str,
                        maximum_elements=None,
                    )
        elif type(object_name) == list and (type(mesh_size) == float or type(mesh_size) == int):
            mesh_str = f"{mesh_size} mm"
            self.maxwell_3d.mesh.assign_length_mesh(
                assignment=object_name,
                inside_selection=inside,
                maximum_length=mesh_str,
                maximum_elements=None,
            )
        else:
            raise ValueError(f"Please recheck all the object name and its corresponding mesh size")
        self.maxwell_3d.eddy_effects_on(object_name)
        return True

    def skin_depth_mesh(self,
                        object_name: list,
                        skin_depth_size: float = 0.2,
                        layer_number: int = 2,
                        mesh_size: float = 0.1,
                        **kwargs):
        """Set up the skindepth based mesh size of the object.

        This function allows users to assign the skin depth based mesh size of the object in Maxwell 3D.
        The mesh size is determined by the user. Be aware that the object name should be the existed object
        in the project and in list form.

        Parameters:
        ----------
        object_name: list,
            The name of the object need assign mesh size. The object name should already existed and
            in list way.
        skin_depth_size: float,
            The skin depth size of the object. The default is 0.2 mm. The unit is mm.
        layer_number: int,
            The mesh layer that for skindepth based mesh. The default is 2.
        mesh_size: float,
            The maximum mesh size of the object. The default is 0.1 mm. if the object name is a list,
            the mesh size should be a list with the same length as the object name list.

        Return:
        -------
        True: bool,
        If the mesh assignment is successful, return True.

        Example:
        --------
        After initialize the project and create a specimen, assign the skin depth mesh size of the
        specimen to 0.1 mm.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_c', design_name='test_design_c')
        >>> m3d = sim.simulation_init()
        >>> _ = sim.create_specimen()
        >>> sim.skin_depth_mesh(object_name=['MySpecimen'], mesh_size=0.1)
        True
        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if type(object_name) == list:
            for obj in object_name:
                if obj not in self._coil_names and obj not in self._specimen_name:
                    raise ValueError(f"Object name: '{obj}' not found. Please recheck the object name.")
            skin_depth = f'{skin_depth_size} mm'
            size = f'{mesh_size} mm'
            mesh_layer = str(layer_number)
            self.maxwell_3d.mesh.assign_skin_depth(
                assignment=object_name,
                skin_depth=skin_depth,
                triangulation_max_length=size,
                layers_number=mesh_layer,
            )
        else:
            raise ValueError(f"Please recheck all the object name and its corresponding mesh size")
        self.maxwell_3d.eddy_effects_on(object_name)
        return True

    def region_assign(self,
                      is_percent: bool = True,
                      boundaries_size: float or int or list = 100,
                      **kwargs):
        """Set up the simulation boundary.

        Setup the simulation boundary. The boundaries size is the size of the simulation boundaries and
        boundary condition is nature boundary if solver type is transient and is radiation boundary if
        solver type is EddyCurrent.

        Parameters:
        ----------
        is_percent: bool,
            Region definition in percentage or absolute value. The default is True. The unit is (%) or (mm).
        boundaryies_size: float or list,
            the boundary size of the simulation. The default is 100. The unit is %. Which means boundary in
            all directions are 100 %. If the set to the list, the length of the list should be 6.
            The order of the list is [x_pos, y_pos, z_pos, x_neg, y_neg, z_neg]. The unit depends on
            the is_percent parameter.

        Return:
        -------
        True: bool,
        If the mesh assignment is successful, return True.

        Example:
        --------
        After initialize the project and create a specimen, the boundary is set to 100 % of all direction.

        >>> from model.simulation_loader import SimulationInit
        >>> sim = SimulationInit(project_name='test_project_d', design_name='test_design_d')
        >>> m3d = sim.simulation_init()
        >>> _ = sim.create_specimen()
        >>> sim.region_assign()
        True
        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if type(boundaries_size) == float or type(boundaries_size) == int:
            region = self.maxwell_3d.modeler.create_air_region(x_pos=boundaries_size,
                                                               y_pos=boundaries_size,
                                                               z_pos=boundaries_size,
                                                               x_neg=boundaries_size,
                                                               y_neg=boundaries_size,
                                                               z_neg=boundaries_size,
                                                               is_percentage=is_percent)
        elif type(boundaries_size) == list:
            if len(boundaries_size) != 6:
                raise ValueError(f"Please set the correct boundary size")
            else:
                region = self.maxwell_3d.modeler.create_air_region(x_pos=boundaries_size[0],
                                                                   y_pos=boundaries_size[1],
                                                                   z_pos=boundaries_size[2],
                                                                   x_neg=boundaries_size[3],
                                                                   y_neg=boundaries_size[4],
                                                                   z_neg=boundaries_size[5],
                                                                   is_percentage=is_percent)
        else:
            raise ValueError(f"Please recheck the the boundary size")
        if self.solution_type == "EddyCurrent":
            self.maxwell_3d.assign_radiation(region)
        elif self.solution_type == "Transient":
            pass
        self._boundary.append("Region")
        return True


class Analysis:
    """Initialize the analysis setup after all 3D models are created.

    This class allows users to set up the analysis in Maxwell 3D.

    """

    def __init__(self, sim_init: SimulationInit = None):
        self.sim_init = sim_init or self._get_latest_sim_init()
        self._setup_names = []

    def _get_latest_sim_init(self):
        if not SimulationInit._instances:
            raise RuntimeError("Please initiate a SimulationInit instance first.")
        return SimulationInit._instances[-1]

    def transient_analysis(self,
                           setup_name: str = "MySetup",
                           stop_time: float = 1,
                           time_step: float = 0.1,
                           n_steps: int = 1,
                           steps_from: float = 0,
                           steps_to: float = 1,
                           pre_save_file=True,
                           pre_stop=False,
                           save_simulation=False,
                           sole_solve=True,
                           cores: int = 4,
                           tasks: int = 1,
                           use_auto_settings: bool = True,
                           **kwargs):
        """Transient analysis in Maxwell 3D.

        This function allows users to set up the transient analysis in Maxwell 3D. The transient analysis
        is determined by the user. Be aware that the solver type should be 'Transient'. The stop time is
        the total time of the simulation. The time step is the time interval between two adjacent steps.
        The n_step is save fields every n steps,steps_from is the start time to save field, and steps_to is
        the end time of save field. All time related unit are us. Please refer the Ansys Maxwell Help
        document see more details.

        Parameters:
        ----------
        setup_name: str,
            The name of the analysis setup. The default is 'MySetup'. If the name already exists, a ValueError
            will be raised.
        stop_time: float,
            The total time of the simulation. The default is 1 us. Stop time should be greater than the time step.
        time_step: float,
            The time interval between two adjacent steps. The default is 0.1 us. Time step should be greater than 0.
        n_steps: int,
            Every n steps to save the field. The default is 1.
        steps_from: float,
            The time begin to save fields. The default is 0 us.
        steps_to: float,
            End time of saving fields. The default is 1 us.
        pre_save_file: bool,
            Whether save the 3D model before simulation. The default is True.
        pre_stop: bool,
            Whether the user only want to build a 3D model without simulation. The default is False,
            If pre_save_file set True and pre_stop set also True, that means only save the 3D model
            and do not start simulation.
        save_simulation: bool,
            Whether save the simulation result. The default is False.
        sole_solve: bool,
            Whether to solve the simulation with the current setup and optimetrics setup together.
            The default is True. Which means solve current setup first and then solve the optimetrics setup
            if exists. If set to False, the simulation will wait the optimetrics correct setup and solve together.
        cores: int,
            How many CPU cores to use for the simulation. The default is 4.
        tasks: int,
            How many tasks to run in parallel. The default is 1.
        use_auto_settings: bool,
            Whether to use the auto settings for the simulation. The default is True.

        Return:
        -------
        True: bool,
            If the transient analysis setup is successful, return True.

        Example:
        --------
        After initialize the project, create a transient analysis setup named 'MySetup' with stop time 1 us,
        time step 0.1 us, save fields every 1 step, save fields from 0 us to 1 us, only save 3D model without
        simulation and don't save the simulation result.

        >>> from model.simulation_loader import SimulationInit, Analysis
        >>> sim = SimulationInit(project_name='test_project_e', design_name='test_design_e', solver_type='Transient')
        >>> _ = sim.simulation_init()
        >>> _ = sim.region_assign(is_percent=True, boundaries_size=100)
        >>> analysis = Analysis(sim_init=sim)
        >>> analysis.transient_analysis(setup_name='MySetup', stop_time=1, time_step=0.1,
        ... n_steps=1, steps_from=0, steps_to=1, pre_save_file=True, pre_stop=True, save_simulation=False)
        True
        """

        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if 'Region' not in self.sim_init._boundary:
            raise ValueError(f"mesh or boundary not assigned, please recheck")
        if self.sim_init.solution_type != 'Transient':
            raise ValueError(f"Solver type must be 'Transient', but got '{self.sim_init.solution_type}'.")
        if setup_name in self._setup_names:
            raise ValueError(f"setup name: '{setup_name}' already existed. Please using another name.")
        self._setup_names.append(setup_name)
        if stop_time <= time_step:
            raise ValueError("The stop time must be greater than the time step.")
        if time_step <= 0:
            raise ValueError("The time step must be greater than 0.")
        if n_steps <= 0:
            raise ValueError("The n_steps must be greater than 0.")
        if steps_from >= steps_to:
            raise ValueError("The steps_from must be less than steps_to.")
        if steps_from < 0:
            raise ValueError("The steps_from must be greater than 0.")
        stoptime = f'{stop_time} us'
        timestep = f'{time_step} us'
        stepsfrom = f'{steps_from} us'
        stepsto = f'{steps_to} us'
        m3d = self.sim_init.maxwell_3d
        setup = m3d.create_setup(name=setup_name)
        setup.props["StopTime"] = stoptime
        setup.props["TimeStep"] = timestep
        setup.props["SaveFieldsType"] = "Every N Steps"
        setup.props["N Steps"] = n_steps
        setup.props["Steps From"] = stepsfrom
        setup.props["Steps To"] = stepsto
        setup.update()
        if pre_save_file == True:
            m3d.save_project()
        else:
            pass
        if pre_stop == True:
            m3d.release_desktop()
            self.sim_init.logger.info("3D model sucessfully build, end of program")
            sys.exit(0)
        else:
            if len(self.sim_init._variable_names) == 3:
                m3d.analyze(cores=cores, tasks=tasks, use_auto_settings=use_auto_settings)
            elif len(self.sim_init._variable_names) != 3 and sole_solve == True:
                m3d.analyze(cores=cores, tasks=tasks, use_auto_settings=use_auto_settings)
            elif len(self.sim_init._variable_names) != 3 and sole_solve == False:
                self.sim_init.logger.info("Variable names already exist, waiting for optimetrics setup")
        if save_simulation == True:
            m3d.save_project()
        else:
            pass
        return True

    def ec_analysis(self,
                    setup_name: str = "MySetup",
                    frequency: float or str = 1.0,
                    percent_error: float = 0.1,
                    pre_save_file=True,
                    pre_stop=False,
                    save_simulation=False,
                    sole_solve=True,
                    cores: int = 4,
                    tasks: int = 1,
                    use_auto_settings: bool = True,
                    **kwargs,
                    ):
        """Eddy current analysis in Maxwell 3D.

        This function allows users to set up the EddyCurrent analysis in Maxwell 3D. The EddyCurrent analysis
        is determined by the user. Be aware that the solver type should be 'EddyCurrent'. The frequency is
        the excitation frequency. More details please refer the Ansys Maxwell Help document.

        Parameters:
        ----------
        setup_name: str,
            The name of the analysis setup. The default is 'MySetup'. If the name already exists, a ValueError
            will be raised.
        frequency: float,
            The excitation frequency of the simulation. The default is 1 MHz.
        pre_stop: bool,
            Whether the user only want to build a 3D model without simulation. The default is False,
            If pre_save_file set True and pre_stop set also True, that means only save the 3D model
            and do not start simulation.
        save_simulation: bool,
            Whether save the simulation result. The default is False.
        sole_solve: bool,
            Whether to solve the simulation with the current setup and optimetrics setup together.
            The default is True. Which means solve current setup first and then solve the optimetrics setup
            if exists. If set to False, the simulation will wait the optimetrics correct setup and solve together.
        cores: int,
            How many CPU cores to use for the simulation. The default is 4.
        tasks: int,
            How many tasks to run in parallel. The default is 1.
        use_auto_settings: bool,
            Whether to use the auto settings for the simulation. The default is True.

        Return:
        -------
        True: bool,
            If the EddyCurrent analysis setup is successful, return True.

        Example:
        --------

        After initialize the project, create an EddyCurrent analysis setup named 'MySetup' with excitation
        frequency 1 MHz, just save 3D model.

        >>> from model.simulation_loader import SimulationInit, Analysis
        >>> sim = SimulationInit(project_name='test_project_f', design_name='test_design_f', solver_type='EddyCurrent')
        >>> _ = sim.simulation_init()
        >>> _ = sim.region_assign(is_percent=True, boundaries_size=100)
        >>> analysis = Analysis(sim_init=sim)
        >>> analysis.ec_analysis(setup_name='MySetup', frequency=1, pre_save_file=True, pre_stop=True, save_simulation=False)
        True

        """
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if 'Region' not in self.sim_init._boundary:
            raise ValueError(f"mesh or boundary not assigned, please recheck")
        if self.sim_init.solution_type != 'EddyCurrent':
            raise ValueError(f"Solver type must be 'EddyCurrent', but got '{self.sim_init.solution_type}'.")
        if setup_name in self._setup_names:
            raise ValueError(f"setup name: '{setup_name}' already existed. Please using another name.")
        if type(frequency) == str:
            if not any(frequency in var_dict for var_dict in self.sim_init._variable_names):
                raise ValueError(f"variable '{frequency}' not found. Please recheck the variable name")
        self._setup_names.append(setup_name)
        if type(frequency) == float:
            Freq = f'{frequency} MHz'
        elif type(frequency) == str:
            Freq = '$' + frequency
        m3d = self.sim_init.maxwell_3d
        setup = m3d.create_setup(name=setup_name)
        setup.props["Frequency"] = Freq
        setup.props["PercentRefinement"] = 15
        setup.props["MaximumPasses"] = 500
        setup.props["PercentError"] = percent_error
        setup.update()
        if pre_save_file == True:
            m3d.save_project()
        else:
            pass
        if pre_stop == True:
            m3d.release_desktop()
            self.sim_init.logger.info("3D model sucessfully build, end of program")
            sys.exit(0)
        else:
            if len(self.sim_init._variable_names) == 3:
                m3d.analyze(cores=cores, tasks=tasks, use_auto_settings=use_auto_settings)
            elif len(self.sim_init._variable_names) != 3 and sole_solve == True:
                m3d.analyze(cores=cores, tasks=tasks, use_auto_settings=use_auto_settings)
            elif len(self.sim_init._variable_names) != 3 and sole_solve == False:
                self.sim_init.logger.info("Variable names already exist, waiting for optimetrics setup")
        if save_simulation == True:
            m3d.save_project()
        else:
            pass
        return True

    def optimetrics_setup(self,
                          file_name: str = None,
                          variable_name: str = "x",
                          save_field: bool = True,
                          start_value: float = 0,
                          end_value: float = 1,
                          step: int = 10,
                          step_type: str = "LinearCount",
                          pre_save_file=True,
                          pre_stop=False,
                          save_simulation=False,
                          cores: int = 4,
                          tasks: int = 1,
                          use_auto_settings: bool = True,
                          **kwargs):
        """Optimetrics setup in Maxwell 3D.

        This function allows users to set up the optimetrics in Maxwell 3D. Be aware that the optimetrics setup
        should follow the previous analysis setup. The variable name should be the same as the variable name
        that already existed in the project. Also, if choose file to pass variable data, the file should be
        a csv file. The file structure should be like this(noticed adding '$' before variable name):
        * $variable_1, $variable_2, ..., $variable_n
        1 0.1mm, 0.2mm, ..., 0.3mm
        2 1mm, 4mm, ..., 3mm
        ...

        Cautions:
        --------
        1. The lowercase characters x, y, z are default parameters, specifically referring to the specimen's
        movement direction in the global coordinate system during testing.
        2. Current only support adding one optimetrics_setup.

        Parameters:
        ----------
        file_name: str,
            The name of the file that contains the variable data. If None, the variable will be created
            with the given start value, end value, step, and step type. The default is None.
        variable_name: str,
            The name of the variable. The default is '$x'. The variable name should predefined in the
            function create_variable in SimulationInit class.
        save_field: bool,
            Whether save the field data in the optimetrics setup. The default is True.
        start_value: float,
            The start value of the variable. The default is 0.
        end_value: float,
            The end value of the variable. The default is 1.
        step: int,
            The number of steps for the variable. The default is 10.
        step_type: str,
            The type of the step. The default is 'LinearCount'. The step type can be 'LinearCount',
             “LinearStep”, “LogScale”, “SingleValue”.
        pre_save_file: bool,
            Whether save the 3D model before simulation. The default is True.
        pre_stop: bool,
            Whether the user only want to build a 3D model without simulation. The default is False,
            If pre_save_file set True and pre_stop set also True, that means only save the 3D model,
            analysis setup and do not start simulation.
        save_simulation: bool,
            Whether save the simulation result. The default is False.
        cores: int,
            How many CPU cores to use for the simulation. The default is 4.
        tasks: int,
            How many tasks to run in parallel. The default is 1.
        use_auto_settings: bool,
            Whether to use the auto settings for the simulation. The default is True.

        Return:
        -------
        True: bool,
            If the optimetrics setup is successful, return True.

        Example:
        --------
        After initialize the project, create an optimetrics setup with variable name 'MyVariable',
        start value 0, end value 1, step 10, step type 'LinearCount', save all the files.

        >>> from model.simulation_loader import SimulationInit, Analysis, GetResult
        >>> sim = SimulationInit(project_name='test_project_g', design_name='test_design_g', solver_type='EddyCurrent')
        >>> _ = sim.simulation_init()
        >>> _ = sim.create_project_variable(variable_name='MyVariable', variable_value=0.1, variable_unit="mm")
        >>> _ = sim.create_helmholtz_coils(name='HeatCoil', material='Copper', center=(1, 0, 1))
        >>> _ = sim.ec_type_assign_excitation(coil_name='HeatCoil', conductors_number=100, resistance=50, amplitude=10)
        >>> _ = sim.length_mesh(object_name='HeatCoil', mesh_size=1)
        >>> _ = sim.region_assign()
        >>> analysis = Analysis(sim_init=sim)
        >>> _ = analysis.ec_analysis(setup_name='MySetup', frequency=1, pre_save_file=True, pre_stop=False, save_simulation=True)
        >>> analysis.optimetrics_setup(variable_name='MyVariable', start_value=0, end_value=1, step=2,
        ... step_type='LinearCount', pre_save_file=True, pre_stop=False, save_simulation=True)
        True
        """
        if kwargs:
            invalid_args = ", ".join(f"'{k}'" for k in kwargs)
            raise TypeError(
                f"Invalid arguments {invalid_args}. Please recheck the arguments. "
            )
        if 'Region' not in self.sim_init._boundary:
            raise ValueError(f"mesh or boundary not assigned, please recheck")
        if self._setup_names == 0:
            raise ValueError("Please create a setup first.")
        if self.sim_init._variable_names == 0:
            raise ValueError("Please create a variable first.")
        if variable_name not in self.sim_init._variable_names:
            raise ValueError(f"Variable name: '{variable_name}' not found. Please recheck the variable name.")
        m3d = self.sim_init.maxwell_3d
        m3d.modeler.set_working_coordinate_system("Global")
        initial_position = ["$x", "$y", "$z"]
        if len(self.sim_init._specimen_name) != 0:
            m3d.modeler.move(self.sim_init._specimen_name[0], initial_position)
        if file_name is None:
            variable_str = f"${variable_name}"
            opt = m3d.parametrics.add(variable=variable_str, start_point=start_value, end_point=end_value, step=step,
                                      variation_type=step_type, solution=self._setup_names[0], name="sweep")
            opt.props["ProdOptiSetupDataV2"]["SaveFields"] = save_field
            opt.update()
        else:
            file_data = pd.read_csv(file_name)
            header = file_data.columns[1:].tolist()
            for variation in header:
                if variation[1:] not in self.sim_init._variable_names:
                    raise ValueError(f"Variable name: '{variation}' not found. Please recheck the variable name.")
            else:
                m3d.parametrics.add_from_file(file_name, name="sweep")
                opt = m3d.parametrics.setups[0]
                opt.props["ProdOptiSetupDataV2"]["SaveFields"] = save_field
                opt.update()
        if pre_save_file:
            m3d.save_project()
        else:
            pass
        if pre_stop:
            m3d.release_desktop()
            print("optimetrics setup sucessfully build, end of program")
            sys.exit(0)
        else:
            m3d.analyze(cores=cores, tasks=tasks, use_auto_settings=use_auto_settings)
        if save_simulation:
            m3d.save_project()
        else:
            pass
        return True

    def reload_analyze(self,
                       **kwargs):
        m3d = self.sim_init.maxwell_3d.analyze()
        return True


class GetResult(object):
    """Get the result of the simulation.

    This class allows users to create result files after the simulation finished.

    """

    def __init__(self,
                 sim_init: SimulationInit = None,
                 ):
        self.sim_init = sim_init or self._get_latest_sim_init()
        self._setup_names = set()

    def _get_latest_sim_init(self):
        if not SimulationInit._instances:
            raise RuntimeError("Please recheck simulation statues first.")
        return SimulationInit._instances[-1]

    def get_induced_voltage(self,
                            coil_name: str or list = "RecCoil",
                            save_name: str = "InducedVoltage"):
        """Get the induced voltage of the coil.

        This function allows users to create the induced voltage result file after the simulation finished.

        Parameters:
        ----------
        coil_name: str or list,
            The name of coils that need to get the induced voltage. The default is 'RecCoil'.
            Be aware that the coil name should be the same as the coil name that already existed in the project.
        save_name: str,
            The name of the result file. The default is 'InducedVoltage'.

        Return:
        -------
        True: bool,
            If the induced voltage result file is created successfully, return True.

        Example:
        --------
        After finish the simulation, get the induced voltage result file from all receiver coils.

        >>> from model.simulation_loader import SimulationInit, Analysis, GetResult
        >>> sim = SimulationInit(project_name='test_project_h', design_name='test_design_h', solver_type='EddyCurrent')
        >>> _ = sim.simulation_init()
        >>> _ = sim.create_rectangle_coil(name='RecCoil', material='Copper', num_turns=10, center=(1, 0, 1))
        >>> _ = sim.ec_type_assign_excitation('RecCoil')
        >>> _ = sim.region_assign(is_percent=True, boundaries_size=100)
        >>> analysis = Analysis(sim_init=sim)
        >>> _ = analysis.ec_analysis(setup_name='MySetup', frequency=1, save_simulation=False)
        >>> result = GetResult()
        >>> result.get_induced_voltage(coil_name='RecCoil', save_name='InducedVoltage')
        True
        """

        m3d = self.sim_init.maxwell_3d
        post = m3d.post
        expressions = []
        if isinstance(coil_name, list):
            for coil in coil_name:
                if coil not in self.sim_init._coil_names:
                    raise ValueError(f"coil name: '{coil}' not found.")
            winding_name = [obj + '_for_winding' for obj in coil_name]
            expressions = [f"mag(InducedVoltage({name}))" for name in winding_name]
        elif isinstance(coil_name, str):
            if coil_name not in self.sim_init._coil_names:
                raise ValueError(f"coil name: '{coil_name}' not found.")
            expressions = [f"mag(InducedVoltage({coil_name + '_for_winding'}))"]
        else:
            raise TypeError("coil_name must be a list or str.")
        if m3d.solution_type == 'EddyCurrent':
            data = post.get_solution_data(
                domain="Sweep",
                primary_sweep_variable="Freq",
                expressions=expressions,
                context="InducedVoltage")
            data.export_data_to_csv(save_name + ".csv", delimiter=",")
        if m3d.solution_type == 'Transient':
            variations = {f"${item}": ["all"] for item in self.sim_init._variable_names}
            data = post.get_solution_data(
                domain="Sweep",
                primary_sweep_variable="Time",
                expressions=expressions,
                variations=variations,
                context="InducedVoltage")
            data.export_data_to_csv(save_name + ".csv", delimiter=",")
        return True

    def get_mean_B_field(self,
                         object_name: list = ["MySpecimen"],
                         ):
        # notice the unit are predefined in the Ansys Maxwell.
        m3d = self.sim_init.maxwell_3d
        post = m3d.post
        data_list = []
        for name in object_name:
            expression = {
                "name": f"{name}",
                "assignment": name,
                "solution_type": "",
                "assignment_type": ["Solid"],
                "design_type": ["Maxwell 3D"],
                "primary_sweep": "Freq",
                "fields_type": ["Fields"],
                "operations": [
                    "NameOfExpression('<Bx,By,Bz>')",
                    "Operation('Mag')",
                    f"EnterVolume({name})",
                    "Operation('VolumeValue')",
                    "Operation('Mean')",
                ],
                "report": ["Data Table", "Rectangular Plot"],
            }
            add_name = post.fields_calculator.add_expression(expression, f'{name}')
            value = post.fields_calculator.evaluate(add_name)
            dict = {add_name: value}
            data_list.append(dict)
        return data_list
