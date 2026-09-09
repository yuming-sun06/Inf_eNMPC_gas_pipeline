import copy
import json

import pyomo.environ as pyo
from pyomo.contrib.mpc import DynamicModelInterface
from idaes.core.util.model_statistics import degrees_of_freedom

from model import buildNonLinearModel
from utils import (
    css_terminal_last_point,
    dynamic_demand_calculation,
    import_data_from_excel,
    init_network_default,
    set_initial_state_from_css,
)


def make_plant_and_controller_model(
        ocss_file_path, horizon=24, num_time_periods=1,
        periodic_constraints=True, *, network_data_path,
        input_data_path, options_data_path, initial_css_file_path=None,
        dynamic_source_pressure_bar=None):
    """Build the finite controller and one-hour plant with this local model.

    Taking all data paths as explicit keyword arguments prevents the finite
    models from silently using a different installed dataset.
    """
    network_data, input_data = import_data_from_excel(
        network_data_path, input_data_path)
    with open(options_data_path, "r") as stream:
        base_options = json.load(stream)

    solver = pyo.SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    solver.options["tol"] = 1e-4

    # Steady state used to initialize both the controller and the plant.
    steady_options = copy.deepcopy(base_options)
    steady_options["dynamic"] = False
    steady_options["T"] = (
        steady_options["T0"] + steady_options["dt"] / 3600)
    m_steady = buildNonLinearModel(
        network_data, input_data, steady_options)
    init_network_default(m_steady, p_default=55e5)
    m_steady.pSource["source_1", :].fix()
    m_steady.pSource["source_2", :].fix()
    m_steady.wSource["source_3", :].fix()
    t0_steady = m_steady.Times.first()

    ########################### 1st solve: m_steady ###########################
    result = solver.solve(m_steady, tee=True)
    pyo.assert_optimal_termination(result)

    # Six-hour finite prediction model.
    controller_options = copy.deepcopy(base_options)
    controller_options["dynamic"] = True
    controller_options["T"] = horizon
    m_controller = buildNonLinearModel(
        network_data, input_data, controller_options)
    init_network_default(m_controller, p_default=55e5)
    for source in ("source_1", "source_2"):
        source_pressure = (
            float(dynamic_source_pressure_bar)
            if dynamic_source_pressure_bar is not None
            else pyo.value(m_controller.pSource[source, 0])
        )
        m_controller.pSource[source, :].fix(source_pressure)
    source_flow = pyo.value(m_controller.wSource["source_3", 0])
    m_controller.wSource["source_3", :].fix(source_flow)
    t0_controller = m_controller.Times.first()
    if initial_css_file_path is not None:
        set_initial_state_from_css(m_controller, initial_css_file_path)
    else:
        for pipe, vol in m_controller.Pipes_VolExtrR_interm:
            m_controller.interm_p[pipe, vol, t0_controller].set_value(
                pyo.value(m_steady.interm_p[pipe, vol, t0_steady]))
        m_controller.interm_p[:, :, t0_controller].fix()
        for station in m_controller.Stations:
            m_controller.compressor_P[station, t0_controller].set_value(
                pyo.value(m_steady.compressor_P[station, t0_steady]))
        m_controller.compressor_P[:, t0_controller].fix()

    ########################### 2nd solve: m_dyn constant demand ###########################
    result = solver.solve(m_controller, tee=True)
    pyo.assert_optimal_termination(result)
    demand = dynamic_demand_calculation(
        m_controller, num_time_periods=num_time_periods)
    for index in m_controller.wCons:
        if index[0].startswith(("sink", "exit")):
            for counter, t in enumerate(m_controller.Times):
                m_controller.wCons[index[0], index[1], t].fix(
                    demand[index[0]][counter])
    if periodic_constraints:
        css_terminal_last_point(
            m_controller, horizon=horizon,
            ocss_file_path=ocss_file_path)
    # solver.options["mu_init"] = 1e-6
    # solver.options["bound_push"] = 1e-6

    ########################### 3rd solve: m_dyn periodic=False ###########################
    result = solver.solve(m_controller, tee=True)
    pyo.assert_optimal_termination(result)

    # One-hour plant, built from exactly the same local equations and data.
    plant_options = copy.deepcopy(base_options)
    plant_options["dynamic"] = True
    plant_options["T"] = 1
    m_plant = buildNonLinearModel(network_data, input_data, plant_options)
    init_network_default(m_plant, p_default=55e5)

    # Use the controller's already-solved first finite element as the plant's
    # complete initialization trajectory.  This keeps pressure, demand, flow,
    # density, and compressor values consistent at every plant collocation
    # point while retaining the 41 bar CSS state at t=0.
    controller_interface = DynamicModelInterface(
        m_controller, m_controller.Times)
    plant_interface = DynamicModelInterface(m_plant, m_plant.Times)
    plant_times = list(m_plant.Times)
    initial_plant_data = controller_interface.get_data_at_time(plant_times)
    shared_variables = (
        [m_controller.compressor_P[s, :] for s in m_controller.Stations]
        + [m_controller.compressor_beta[s, :] for s in m_controller.Stations]
        + [
            m_controller.wCons[n, v, :]
            for n, v in {
                (index[0], index[1]) for index in m_controller.wCons
            }
        ]
        + [m_controller.node_p[n, :] for n in m_controller.Nodes]
        + [m_controller.interm_w[p, v, :]
           for p, v in m_controller.Pipes_VolExtrC_interm]
        + [m_controller.interm_p[p, v, :]
           for p, v in m_controller.Pipes_VolExtrR_interm]
        + [m_controller.wSource[s, :] for s in m_controller.NodesSources]
        + [m_controller.pSource[s, :] for s in m_controller.NodesSources]
        + [m_controller.pipe_rho[p, v, :]
           for p, v in m_controller.Pipes_VolExtrR]
        + [m_controller.inlet_w[a, :] for a in m_controller.Arcs]
        + [m_controller.outlet_w[a, :] for a in m_controller.Arcs]
        + [m_controller.u[p, v, :]
           for p, v in m_controller.Pipes_VolExtrC]
        + [m_controller.u2[p, v, :]
           for p, v in m_controller.Pipes_VolExtrC]
    )
    initial_plant_data = initial_plant_data.extract_variables(
        shared_variables)
    plant_interface.load_data(initial_plant_data)

    m_plant.pSource["source_1", :].fix()
    m_plant.pSource["source_2", :].fix()
    m_plant.wSource["source_3", :].fix()
    t0_plant = m_plant.Times.first()
    m_plant.interm_p[:, :, t0_plant].fix()
    for station in m_plant.Stations:
        for t in m_plant.Times:
            m_plant.compressor_P[station, t].fix(
                pyo.value(m_controller.compressor_P[station, t]))
    m_plant.ObjFun.deactivate()
    if degrees_of_freedom(m_plant) != 0:
        raise RuntimeError(
            f"Local plant initialization has "
            f"{degrees_of_freedom(m_plant)} degrees of freedom")

    ########################### 4th solve: plant ###########################
    result = solver.solve(m_plant, tee=True)
    pyo.assert_optimal_termination(result)
    return m_controller, m_plant
