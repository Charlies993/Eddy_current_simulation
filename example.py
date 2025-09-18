# -*- coding: utf-8 -*-
"""
@Project :  AI_digital_twin
@Product :  PyCharm
@File    :  test.py
@Time    :  2025/2/26 9:54
@Author  :  Chang_Chao
@Contact :  changchao993@163.com
@Desc    :  simulation test
"""


from model.simulation_loader import SimulationInit, Analysis, GetResult

# Here we create an eddy current solver (frequency domain) simulation.
# The project name is 'My_project' and the design name is 'My_design'.
sim = SimulationInit(project_name='My_project', design_name='My_design', solver_type='EddyCurrent')

# We initialize the simulation.
sim.simulation_init()

# Here we create project variables for optimitrics setup.
sim.create_project_variable(variable_name="h", variable_value=0, variable_unit="mm")

# Here we create a spiral coil named e00, the total number of turns is 16, the wire width is 0.125 mm,
# the wire height is 0.035 mm, the spacing between turns is 0.75 mm, and the inner radius of the coil is 2.5 mm.
# The coil position is set to (0 mm, 0 mm, 5 mm) at the global coordinate system.
sim.create_spiral_coil("e00", num_turns=16, wire_width=0.125, wire_height=0.035, spacing=0.25,
                       coil_inner_radius=2.15, center=(16.5, 16.5, 1))

# Here we assign the excitation resource to the coil e00.
# The excitation type is voltage, the voltage is 4 V.
# We want load all the voltage to the coil, thus we set the impendance to 0.001 ohm.
sim.ec_type_assign_excitation("e00", conductors_number=1, amplitude=4.0, resistance=0.001)

# Here we create another excitation rectangle coil named e11, the total number of turns is 20,
# the wire width is 0.125 mm, the wire height is 0.035 mm, the spacing between turns is 0.75 mm,
# and the initial x length and y length are both 2.5 mm. The position is set to (0 mm, 0 mm, 10 mm).
sim.create_rectangle_coil("e11", num_turns=20, wire_width=0.125, wire_height=0.035, step_size=0.75,
                          initial_x_length=2.5, initial_y_length=2.5, center=(0, 0, 10))

# Here we assign the excitation resource to the coil e11.
# The excitation type is voltage, the voltage is 4 V.
# We want load all the voltage to the coil, thus we set the impendance to 0.001 ohm.
sim.ec_type_assign_excitation("e11", conductors_number=1, amplitude=4.0, resistance=0.001)

# Here we create a Helmholtz receiver coil named r00. the inner diameter is 2.5 mm, the outer diameter is 3.5 mm,
# and the position is set to (0 mm, 0 mm, 2 mm). The coil height is 0.25 mm.
sim.create_helmholtz_coils("r00", inner_diameter=2.5, center=(0, 0, 2), height=0.25)

# Here we assign a 0 V voltage to the receiver coil r00 to perform a receiver. We assume that the helmholtz coil
# has total 100 turns, we further want to load all the induced voltage to the coil, thus we set the impedance
# to 5e6 ohm to perform a break circuit.
sim.ec_type_assign_excitation("r00", conductors_number=100, amplitude=0.0, resistance=5e6)

# Here we create a specimen named 'My_specimen' with a size of 33 mm x 33 mm x 10 mm. The material is aluminum.
sim.create_specimen(specimen_name="My_specimen", material="aluminum", length=33, width=33, height=10)

# Here we sequence assign the defects to the specimen.
# The first defect length/width/height are all 0.5 mm,
# located at (0 mm, 0 mm, 0 mm) in global coordinate system.
sim.specimen_with_crack(specimen_name="My_specimen", crack_length=0.5,
                        crack_width=0.5, crack_height=0.5, center=(0, 0, 0))

# Here we create the second defect with length/width/height is 0.1mm, 1 mm, 0.5 mm respectively,
# located at (0 mm, 0 mm, -1 mm) in global coordinate system.
sim.specimen_with_crack(specimen_name="My_specimen", crack_length=0.1,
                        crack_width=1, crack_height=0.5, center=(0, 0, -1))

# Here we assign a 0.1 mm mesh size to e00, 0.2 mm mesh size to e11, and 0.5 mm mesh size to r00
sim.length_mesh(["e00", "e11", "r00"], mesh_size=[0.1, 0.2, 0.5])

# Here we create a region for simulation. The region is expanded by 10 mm in all directions.
sim.region_assign(is_percent=False, boundaries_size=10)

# We initialize the analyze module.
analysis = Analysis()

# Here we create a analysis setup named 'My_setup' with a frequency of 1 MHz. We only want build the
# 3D model without solving the simulation, thus we set the pre_save_file to True and
# pre_stop to Ture while save_simulation to False.
analysis.ec_analysis(setup_name="My_setup",pre_save_file=False, pre_stop=False, save_simulation=False)

# Here we create optimetrics setup for the simulation.
analysis.optimetrics_setup(variable_name="h", start_value=0, end_value=10, step=1, pre_stop=True)

# If we set the pre_stop to False, Then the following code will continue work. Else, the simulation and
# the code process will stop here.
# Here we initialize the post processing module to get the desired result.
result = GetResult()

# Here we get the induced voltage of the receiver coil r00, and save as csv file.
result.get_induced_voltage("e00")
