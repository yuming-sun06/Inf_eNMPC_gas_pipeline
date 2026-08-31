import json
import os
import time

import pyomo.contrib.mpc as mpc
import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.common.timing import HierarchicalTimer

from model import buildNonLinearModel, init_network_default
from run_model import run_model
from terminal import load_css, update_controller_obj
from utils import (
    Tee,
    dynamic_demand_calculation,
    import_data_from_excel,
    write_data_to_excel,
)

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def add_mpc_soft_terminal_constraint(m):
    """Add the terminal-pressure slack after non-periodic warm-up."""
    tf = m.Times.last()
    t0 = m.Times.first()

    m.terminal_p_slack = pyo.Var(
        m.Pipes_VolExtrR_interm,
        initialize=0,
        domain=pyo.Reals,
    )

    def _terminal_pipe_pressure(model, pipe, vol):
        return (
            model.interm_p[pipe, vol, tf]
            == model.interm_p_ocss[pipe, vol, t0]
            + model.terminal_p_slack[pipe, vol]
        )

    m.terminal_pipe_pressure = pyo.Constraint(
        m.Pipes_VolExtrR_interm,
        rule=_terminal_pipe_pressure,
    )
    return m


#==============================================================================
""" plant_data """
#==============================================================================

def get_plant_data(
        network_data_path=None,
        input_data_path=None,
        options_data_path=None,
        plant_horizon=1):
    if network_data_path is None:
        network_data_path = os.path.join(_BASE_DIR, "Input_data", "networkData.xlsx")
    if input_data_path is None:
        input_data_path = os.path.join(
            _BASE_DIR,
            "Input_data",
            "inputData_longer_horizon.xlsx",
        )
    if options_data_path is None:
        options_data_path = os.path.join(_BASE_DIR, "Input_data", "Options.json")

    network_data, input_data = import_data_from_excel(
        network_data_path,
        input_data_path,
    )
    with open(options_data_path, "r") as file:
        options = json.load(file)

    options["dynamic"] = True
    options["T"] = plant_horizon
    return network_data, input_data, options


#==============================================================================
""" controller_and_plant """
#==============================================================================

def make_controller_and_plant(
        ocss_file_path,
        initial_css_file_path,
        controller_horizon=24,
        plant_horizon=1,
        num_time_periods=1,
        warmup_periodic=True,
        soft=False):
    _, m_controller = run_model(
        horizon=controller_horizon,
        num_time_periods=num_time_periods,
        ocss_file_path=ocss_file_path,
        initial_css_file_path=initial_css_file_path,
        periodic_constraints=warmup_periodic,
        soft=soft,
    )
    if not warmup_periodic:
        # The non-periodic warm-up does not create the fixed OCSS reference
        # variables. Load them separately for the MPC terminal condition.
        load_css(m_controller, ocss_file_path)

    network_data, input_data, options = get_plant_data(
        plant_horizon=plant_horizon,
    )
    m_plant = buildNonLinearModel(network_data, input_data, options)
    m_plant = init_network_default(m_plant, p_default=55e5)

    # Initialize the plant from the controller's already-solved trajectory over
    # the same time points. This keeps t=0 at the 41 bar CSS state while making
    # the t=1 pressure, demand, flow, density, and compressor values mutually
    # consistent with the controller's first predicted transition.
    controller_interface = mpc.DynamicModelInterface(
        m_controller, m_controller.Times
    )
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.Times)
    plant_times = list(m_plant.Times)
    initial_plant_data = controller_interface.get_data_at_time(plant_times)
    shared_plant_variables = (
        [m_controller.compressor_P[s, :] for s in m_controller.Stations]
        + [m_controller.compressor_beta[s, :] for s in m_controller.Stations]
        + [
            m_controller.wCons[n, v, :]
            for n, v in {
                (index[0], index[1]) for index in m_controller.wCons
            }
        ]
        + [m_controller.node_p[n, :] for n in m_controller.Nodes]
        + [m_controller.interm_w[p, v, :] for p, v in m_controller.Pipes_VolExtrC_interm]
        + [m_controller.interm_p[p, v, :] for p, v in m_controller.Pipes_VolExtrR_interm]
        + [m_controller.wSource[s, :] for s in m_controller.NodesSources]
        + [m_controller.pSource[s, :] for s in m_controller.NodesSources]
        + [m_controller.pipe_rho[p, v, :] for p, v in m_controller.Pipes_VolExtrR]
        + [m_controller.inlet_w[a, :] for a in m_controller.Arcs]
        + [m_controller.outlet_w[a, :] for a in m_controller.Arcs]
        + [m_controller.u[p, v, :] for p, v in m_controller.Pipes_VolExtrC]
        + [m_controller.u2[p, v, :] for p, v in m_controller.Pipes_VolExtrC]
    )
    initial_plant_data = initial_plant_data.extract_variables(
        shared_plant_variables
    )
    plant_interface.load_data(initial_plant_data)

    t0_plant = m_plant.Times.first()
    m_plant.pSource["source_1", :].fix()
    m_plant.pSource["source_2", :].fix()
    m_plant.wSource["source_3", :].fix()

    m_plant.interm_p[:, :, t0_plant].fix()

    for station in m_plant.Stations:
        for t in m_plant.Times:
            m_plant.compressor_P[station, t].fix(
                pyo.value(m_controller.compressor_P[station, t])
            )

    m_plant.ObjFun.deactivate()
    assert degrees_of_freedom(m_plant) == 0

    solver = pyo.SolverFactory("ipopt")
    solver.options["linear_solver"] = "ma57"
    print("\n================ plant initialization solve ================")
    print("degrees_of_freedom =", degrees_of_freedom(m_plant))
    result = solver.solve(m_plant, tee=True)
    pyo.assert_optimal_termination(result)

    return m_controller, m_plant


#==============================================================================
""" demand_data """
#==============================================================================

def load_demand_data(m, demand_data, start, stop):
    for sink in m.sink_node_set:
        values = demand_data[sink][int(start):int(stop)]
        for index, t in enumerate(m.Times):
            m.actual_demand[sink, t] = values[index]
            m.wCons[sink, 0, t].fix(m.actual_demand[sink, t])


#==============================================================================
""" run_nmpc """
#==============================================================================

def run_nmpc(
        simulation_steps=24,
        sample_time=1,
        controller_horizon=24,
        plant_horizon=1,
        num_time_periods=1,
        ocss_file_path=None,
        initial_css_file_path=None,
        warmup_periodic=True,
        terminal_penalty_coeff=1e6,
        soft=False,
        metrics_callback=None):
    if ocss_file_path is None:
        ocss_file_path = os.path.join(_BASE_DIR, "css.xlsx")
    if initial_css_file_path is None:
        initial_css_file_path = os.path.join(_BASE_DIR, "css_41.xlsx")

    timer = HierarchicalTimer()
    timer.start("Initialization")
    m_controller, m_plant = make_controller_and_plant(
        ocss_file_path=ocss_file_path,
        initial_css_file_path=initial_css_file_path,
        controller_horizon=controller_horizon,
        plant_horizon=plant_horizon,
        num_time_periods=num_time_periods,
        warmup_periodic=warmup_periodic,
        soft=soft,
    )
    timer.stop("Initialization")
    if soft:
        if not warmup_periodic:
            add_mpc_soft_terminal_constraint(m_controller)
        update_controller_obj(
            m_controller,
            coeff=terminal_penalty_coeff,
        )

    sink_nodes = [n for n in m_controller.Nodes if str(n).startswith("sink")]
    m_controller.sink_node_set = pyo.Set(initialize=sink_nodes)
    m_plant.sink_node_set = pyo.Set(initialize=sink_nodes)
    m_controller.actual_demand = pyo.Param(
        m_controller.sink_node_set,
        m_controller.Times,
        initialize=1,
        mutable=True,
    )
    m_plant.actual_demand = pyo.Param(
        m_plant.sink_node_set,
        m_plant.Times,
        initialize=1,
        mutable=True,
    )

    demand_length = simulation_steps + controller_horizon + 1
    cycle_length = controller_horizon / num_time_periods
    demand_periods = (demand_length - 1) / cycle_length
    demand_data = dynamic_demand_calculation(
        m_controller,
        num_time_periods=demand_periods,
        time_length=demand_length,
    )

    controller_interface = mpc.DynamicModelInterface(
        m_controller,
        m_controller.Times,
    )
    plant_interface = mpc.DynamicModelInterface(m_plant, m_plant.Times)
    controller_t0 = m_controller.Times.first()
    non_initial_plant_time = list(m_plant.Times)[1:]

    plant_controls = [
        m_controller.compressor_P[station, :]
        for station in m_controller.Stations
    ]

    solver = pyo.SolverFactory("ipopt")
    solver.options["tol"] = 1e-4 #---------------------------------------------------------------
    solver.options["linear_solver"] = "ma57"

    sim_data = plant_interface.get_data_at_time([0.0])
    iteration_times = {}

    #==============================================================================
    """ nmpc_loop """
    #==============================================================================

    for iteration in range(simulation_steps):
        print(
            f"\n================ NMPC iteration {iteration} =================",
            flush=True,
        )
        timer.start(f"iteration_{iteration}")
        simulation_time = iteration * sample_time

        load_demand_data(
            m_controller,
            demand_data,
            simulation_time,
            simulation_time + controller_horizon + 1,
        )
        load_demand_data(
            m_plant,
            demand_data,
            simulation_time,
            simulation_time + plant_horizon + 1,
        )

        print("controller degrees_of_freedom =", degrees_of_freedom(m_controller))
        timer.start("Solve_controller")
        controller_result = solver.solve(m_controller, tee=True)
        timer.stop("Solve_controller")
        pyo.assert_optimal_termination(controller_result)
        active_objectives = list(
            m_controller.component_data_objects(pyo.Objective, active=True)
        )
        objective_value = sum(
            pyo.value(objective.expr) for objective in active_objectives
        )
        if metrics_callback is not None:
            metrics_callback({
                "iteration": iteration,
                "objective": objective_value,
            })

        control_data = controller_interface.get_data_at_time(sample_time)
        control_data = control_data.extract_variables(plant_controls)
        plant_interface.load_data(
            control_data,
            time_points=non_initial_plant_time,
        )

        print("plant degrees_of_freedom =", degrees_of_freedom(m_plant))
        assert degrees_of_freedom(m_plant) == 0
        timer.start("Solve_plant")
        plant_result = solver.solve(m_plant, tee=True)
        timer.stop("Solve_plant")
        pyo.assert_optimal_termination(plant_result)

        plant_data = plant_interface.get_data_at_time(non_initial_plant_time)
        plant_data.shift_time_points(simulation_time - m_plant.Times.first())
        sim_data.concatenate(plant_data)

        final_plant_data = plant_interface.get_data_at_time(m_plant.Times.last())
        plant_interface.load_data(final_plant_data)
        controller_interface.shift_values_by_time(sample_time)
        controller_interface.load_data(
            final_plant_data,
            time_points=controller_t0,
        )

        controller_tf = m_controller.Times.last()
        for station in m_controller.Stations:
            m_controller.compressor_P_ocss[station, controller_tf].fix(
                m_controller.compressor_P_ocss[station, controller_t0]
            )
        for p, vol in m_controller.Pipes_VolExtrR_interm:
            m_controller.interm_p_ocss[p, vol, controller_tf].fix(
                m_controller.interm_p_ocss[p, vol, controller_t0]
            )

        timer.stop(f"iteration_{iteration}")
        iteration_times[iteration] = timer.get_total_time(f"iteration_{iteration}")
        print(
            f"Iteration {iteration} time: {iteration_times[iteration]:.2f}s",
            flush=True,
        )

    print(timer)
    return m_plant, m_controller, sim_data, iteration_times


#==============================================================================
""" write_results """
#==============================================================================

def write_results(
        sim_data, m_plant, iteration_times,
        total_runtime, output_path):
    sheets_keys_dict = {
        "compressor power": [
            m_plant.compressor_P[s, :]
            for s in m_plant.Stations
        ],
        "compressor beta": [
            m_plant.compressor_beta[s, :]
            for s in m_plant.Stations
        ],
        "wCons": [
            m_plant.wCons[s, 0, :]
            for s in m_plant.Nodes
            if str(s).startswith("sink")
        ],
        "node pressure": [
            m_plant.node_p[n, :]
            for n in m_plant.Nodes
            if str(n).startswith("sink")
        ],
        "interm_w": [
            m_plant.interm_w[p, vol, :]
            for p, vol in m_plant.Pipes_VolExtrC_interm
        ],
        "interm_p": [
            m_plant.interm_p[p, vol, :]
            for p, vol in m_plant.Pipes_VolExtrR_interm
        ],
        "wSource": [m_plant.wSource[s, :] for s in m_plant.NodesSources],
        "pSource": [m_plant.pSource[s, :] for s in m_plant.NodesSources],
    }
    write_data_to_excel(
        sim_data,
        m_plant,
        sheets_keys_dict,
        output_path,
        iteration_times=iteration_times,
        total_runtime=total_runtime,
    )

#==============================================================================
""" main """
#==============================================================================

def main(
        soft=False,
        warmup_periodic=True,
        terminal_penalty_coeff=1e6):
    start_time = time.time()
    m_plant, _, sim_data, iteration_times = run_nmpc(
        simulation_steps=120,
        sample_time=1,
        controller_horizon=120,
        plant_horizon=1,
        num_time_periods=10,
        warmup_periodic=warmup_periodic,
        terminal_penalty_coeff=terminal_penalty_coeff,
        soft=soft,
    )
    total_runtime = time.time() - start_time
    print(f"Total runtime: {total_runtime:.2f}s", flush=True)

    output_path = os.path.join(_BASE_DIR, "finite-10-cycle_no_V_e4.xlsx")
    write_results(
        sim_data,
        m_plant,
        iteration_times,
        total_runtime,
        output_path,
    )


if __name__ == "__main__":
    soft = True
    warmup_periodic = True # 1 cycle = F, 10 cycle = T
    terminal_penalty_coeff = 1e4 # 1 cycle = 1e6, 10 cycle = 1e4
    log_path = os.path.join(_BASE_DIR, "finite-10-cycle_no_V_e4.log")
    tee = Tee(log_path)
    try:
        print("soft =", soft)
        print("warmup_periodic =", warmup_periodic)
        print("terminal_penalty_coeff =", terminal_penalty_coeff)
        main(
            soft=soft,
            warmup_periodic=warmup_periodic,
            terminal_penalty_coeff=terminal_penalty_coeff,
        )
        print("Log saved to:", log_path)
    finally:
        tee.close()
