import copy
import json

import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom

from model import buildNonLinearModel
from utils import (
    css_terminal_last_point,
    dynamic_demand_calculation,
    import_data_from_excel,
    init_network_default,
    load_css,
)


def set_tracking_objective(
        model, ocss_file_path, coeff_p=1.0, coeff_P=1.0):
    """Replace the economic objective with pressure/power tracking."""
    load_css(model, ocss_file_path)
    include_last = len(model.Times) == 1
    model.tracking_p = pyo.Expression(
        expr=coeff_p*sum(
            (model.interm_p[p, vol, t]
             - model.interm_p_ocss[p, vol, t])**2
            for p, vol in model.Pipes_VolExtrR_interm
            for t in model.Times
            if include_last or t != model.Times.last()
        )
    )
    model.tracking_P = pyo.Expression(
        expr=coeff_P*sum(
            (model.compressor_P[s, t]
             - model.compressor_P_ocss[s, t])**2
            for s in model.Stations
            for t in model.Times
            if include_last or t != model.Times.last()
        )
    )
    model.ObjFun.deactivate()
    model.tracking_obj = pyo.Objective(
        expr=model.tracking_p + model.tracking_P)
    return model


def make_plant_and_controller_model(
        ocss_file_path, horizon=24, num_time_periods=1,
        periodic_constraints=True, *, network_data_path,
        input_data_path, options_data_path,
        unfix_P_at_t0=False, coeff_p=1.0, coeff_P=1.0):
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
    m_steady.pSource["source_1", :].fix(40)
    t0_steady = m_steady.Times.first()
    initial_beta = {
        "compressorStation_1": 1.2,
        "compressorStation_2": 1.2,
        "compressorStation_3": 1.1,
    }
    for station in m_steady.Stations:
        m_steady.compressor_beta[station, t0_steady].fix(
            initial_beta.get(str(station), 1.0))

    set_tracking_objective(
        m_steady, ocss_file_path, coeff_p=coeff_p, coeff_P=coeff_P)

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
    source_pressure = pyo.value(m_controller.pSource["source_1", 0])
    m_controller.pSource["source_1", :].fix(source_pressure)
    t0_controller = m_controller.Times.first()
    for pipe, vol in m_controller.Pipes_VolExtrR_interm:
        m_controller.interm_p[pipe, vol, t0_controller].set_value(
            pyo.value(m_steady.interm_p[pipe, vol, t0_steady]))
    m_controller.interm_p[:, :, t0_controller].fix()
    for station in m_controller.Stations:
        m_controller.compressor_P[station, t0_controller].set_value(
            pyo.value(m_steady.compressor_P[station, t0_steady]))
    if unfix_P_at_t0:
        m_controller.compressor_P[:, t0_controller].unfix()
        print("Initial controller compressor_P at t=0 is unfixed")
    else:
        m_controller.compressor_P[:, t0_controller].fix()

    set_tracking_objective(
        m_controller, ocss_file_path, coeff_p=coeff_p, coeff_P=coeff_P)

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
    solver.options["mu_init"] = 1e-6
    solver.options["bound_push"] = 1e-6

    ########################### 3rd solve: m_dyn periodic=False ###########################
    result = solver.solve(m_controller, tee=True)
    pyo.assert_optimal_termination(result)

    # One-hour plant, built from exactly the same local equations and data.
    plant_options = copy.deepcopy(base_options)
    plant_options["dynamic"] = True
    plant_options["T"] = 1
    m_plant = buildNonLinearModel(network_data, input_data, plant_options)
    init_network_default(m_plant, p_default=55e5)
    m_plant.pSource["source_1", :].fix()
    t0_plant = m_plant.Times.first()
    for pipe, vol in m_plant.Pipes_VolExtrR_interm:
        m_plant.interm_p[pipe, vol, t0_plant].set_value(
            pyo.value(m_steady.interm_p[pipe, vol, t0_steady]))
    m_plant.interm_p[:, :, t0_plant].fix()
    for station in m_plant.Stations:
        initial_power = pyo.value(
            m_controller.compressor_P[station, t0_controller])
        for t in m_plant.Times:
            m_plant.compressor_P[station, t].fix(initial_power)
    m_plant.ObjFun.deactivate()
    if degrees_of_freedom(m_plant) != 0:
        raise RuntimeError(
            f"Local plant initialization has "
            f"{degrees_of_freedom(m_plant)} degrees of freedom")

    ########################### 4th solve: plant ###########################
    result = solver.solve(m_plant, tee=True)
    pyo.assert_optimal_termination(result)
    return m_controller, m_plant
