#Infinite horizon enmpc for gas networks
from pathlib import Path
import time
import numpy as np
import json
import pandas as pd
import pyomo.environ as pyo
from idaes.core.util.model_statistics import degrees_of_freedom
from pyomo.common.timing import HierarchicalTimer
from pyomo.contrib.mpc import DynamicModelInterface
from model import (
    Pipe_MassFlow,
    buildNonLinearModel,
)
import make_controller_plant as controller_plant_builder
from utils import (
    dynamic_demand_calculation,
    import_data_from_excel,
    init_network_default,
    load_css,
    write_data_to_excel,
)

BASE_PATH = Path(__file__).resolve().parent
SEC_PER_HOUR = 3600.0

class ControlGasNet:
    def __init__(self):
        self.Horizon = 12
        self.NumCycles = 1
        self.Sampling_time = 1
        self.Cycle_length = 12
        self.Simulation_period = 120

        self.infinite_horizon = True
        self.unfix_P_at_t0 = True
        self.NFE_infinite_horizon = 3
        self.NCP_infinite_horizon = 3
        self.Gamma = 0.1

        self.coeff_p = 1e-1
        self.coeff_P = 1e-1
        self.coeff_pSource3 = 1e-1
        self.terminal = True
        self.terminal_coeff = 1e6

        #IPOPT
        self.ipopt = pyo.SolverFactory('ipopt')
        self.ipopt.options["tol"] = 1e-4
        self.ipopt.options["linear_solver"] = "ma57"

        # Data path
        self.network_data_path = str(BASE_PATH / "Input_data" / "networkData.xlsx")
        self.ocss_data_path    = str(BASE_PATH / "css.xlsx")
        self.initial_css_data_path = str(BASE_PATH / "css_41.xlsx")
        self.dynamic_source_pressure_bar = 35.01325
        self.input_data_path   = str(BASE_PATH / "Input_data" / "inputData.xlsx")
        self.options_data_path = str(BASE_PATH / "Input_data" / "Options.json")

        # Output path
        self.results_path = str(BASE_PATH / "Gaussian_quadrature.xlsx")
        self.log_path = str(BASE_PATH / "Gaussian_quadrature.txt")

    # ============================================================================
    # Demand loading and horizon updates
    # ============================================================================
    def load_demand_data(self, m, demand_data, start=0, stop=6, soft_constraint=False):
        for s in m.sink_node_set:
            actual_demand = demand_data[s][start:stop]
            for t_index, t in enumerate(m.Times, start=0):
                m.actual_demand[s, t] = actual_demand[t_index]
                if not soft_constraint:
                    m.wCons[s, 0, t].fix(m.actual_demand[s, t])

    def shift_time_indexed_parameters_finite(self, phase_offset):
        model = self.m_controller.finite

        tf = model.Times.last()
        t0 = model.Times.first()

        # Move the parameters by sampling time
        for s in model.sink_node_set:
            for t in model.Times:
                if t != tf:
                    model.actual_demand[s, t] = pyo.value(model.actual_demand[s, model.Times.next(t)])
                    model.wCons[s, 0, t].fix(model.actual_demand[s, t])

            model.actual_demand[s, tf] = pyo.value(model.actual_demand[s, t0])
            model.wCons[s, 0, tf].fix(model.actual_demand[s, tf])

        for tau in self.m_controller.infinite.Times:
            if tau == 1:
                for s in self.m_controller.infinite.sink_node_set:
                    self.m_controller.infinite.wCons[s, :, tau].fix(pyo.value(self.m_controller.finite.wCons[s, 0, tf]))

            else:
                t_bar = self.Horizon
                gamma = pyo.value(self.m_controller_inf.gamma)
                t = phase_offset + t_bar + 1/gamma*np.arctanh(tau)

                sin_func = np.sin(2*np.pi/self.Cycle_length*t)

                for s in self.m_controller.infinite.sink_node_set:
                    node_nominal_demand = self.nominal_demand[s]
                    self.m_controller_inf.wCons[s, 0, tau].fix(
                        node_nominal_demand
                        + node_nominal_demand/20*sin_func)

    # ============================================================================
    # Finite-horizon controller and plant
    # ============================================================================
    def make_controller_and_plant_model_finite(self):
        # Keep compressor_P(t=0) fixed during the finite warm-up. The NMPC
        # loop applies unfix_P_at_t0 immediately before each controller solve.
        self.m_controller_finite, self.m_plant = (
            controller_plant_builder.make_plant_and_controller_model(
                self.ocss_data_path,
                horizon=self.Horizon,
                num_time_periods=self.NumCycles,
                periodic_constraints=False,
                network_data_path=self.network_data_path,
                input_data_path=self.input_data_path,
                options_data_path=self.options_data_path,
                initial_css_file_path=self.initial_css_data_path,
                # Use the same source pressure in the finite warm-up and
                # the transformed infinite tail.
                dynamic_source_pressure_bar=self.dynamic_source_pressure_bar,
                coeff_p=self.coeff_p,
                coeff_P=self.coeff_P,
                coeff_pSource3=self.coeff_pSource3,
            )
        )
        
        # If finite horizon only, no need to load                                                                       
        load_css(self.m_controller_finite, ocss_file_path=self.ocss_data_path)

        # If use rho for endpoint
        rho_css_data = pd.read_excel(
            self.ocss_data_path, sheet_name="pipe_rho", index_col=0)
        self.m_controller_finite.pipe_rho_ocss = pyo.Var(
            self.m_controller_finite.Pipes_VolExtrR_interm,
            self.m_controller_finite.Times)
        for p, vol in self.m_controller_finite.Pipes_VolExtrR_interm:
            column = f"pipe_rho['{p}', {vol}, :]"
            for t in self.m_controller_finite.Times:
                self.m_controller_finite.pipe_rho_ocss[p, vol, t].fix(
                    float(rho_css_data.loc[t, column]))

        # Load demand profile
        sink_node_set = [
            s for s in self.m_controller_finite.Nodes
            if str(s).startswith(("sink", "exit"))
        ]
        self.m_controller_finite.sink_node_set = pyo.Set(initialize = sink_node_set)
        self.m_plant.sink_node_set = pyo.Set(initialize = sink_node_set)
        
        demand_data = dynamic_demand_calculation(self.m_controller_finite, 
                                                 num_time_periods = self.NumCycles, 
                                                 extended_profile=True)
        self.nominal_demand = {
            sink: float(profile[0]) for sink, profile in demand_data.items()
        }

        # Load initial demand into the controller
        self.m_controller_finite.actual_demand = pyo.Param(self.m_controller_finite.sink_node_set, 
                                                    self.m_controller_finite.Times, 
                                                    initialize = 1, 
                                                    mutable = True)
        self.m_plant.actual_demand = pyo.Param(self.m_plant.sink_node_set, 
                                               self.m_plant.Times, 
                                               initialize = 1, 
                                               mutable = True)
        self.load_demand_data(self.m_controller_finite, demand_data, start = 0, stop = self.Horizon+1)
      
        self.load_demand_data(self.m_plant, demand_data, start = 0, stop = self.Sampling_time+1)

        # Controller interface
        self.controller_interface = DynamicModelInterface(self.m_controller_finite, self.m_controller_finite.Times)

        # Plant interface
        self.plant_interface = DynamicModelInterface(self.m_plant, self.m_plant.Times)

        # Variables that will be fixed in the plant
        self.plant_fixed_variables = [
            self.m_controller_finite.compressor_P[s, :]
            for s in self.m_controller_finite.Stations
        ]
        self.m_controller_finite.stage_cost = 1/1000*sum(self.m_controller_finite.compressor_P[s, t]*SEC_PER_HOUR
                                                  for s in self.m_controller_finite.Stations
                                                  for t in self.m_controller_finite.Times
                                                  if t != self.m_controller_finite.Times.last())
        # The local tracking objective has completed all finite-model warm-up
        # solves. The assembled finite+infinite controller creates its own
        # complete tracking objective later, so keep only that one active.
        self.m_controller_finite.ObjFun.deactivate()
        self.m_controller_finite.tracking_obj.deactivate()

    # ============================================================================
    # Infinite-horizon helper functions
    # ============================================================================
    # Rebuild infinite mass balance: gamma*(1 - tau^2) * drho/dtau * V/(N-1) == 3600*(w - w_next - cons)
    def _rebuild_infinite_mass_balance(self):
        m = self.m_controller_inf
        scale_w = 1.0  # local model.py scale["w"]

        m.del_component(m.PIPE_mass_balance)

        def _pipe_mass_balance_infhorizon_hours(m, p, vol, t):
            if t == m.Times.first():
                return pyo.Constraint.Skip
            V = m.Area[p]*m.Length[p]
            N = m.Nvol[p]
            w = Pipe_MassFlow(m, p, vol, t, N)
            w_next = Pipe_MassFlow(m, p, vol+1, t, N)  # next finite volume
            cons = m.wCons[p, vol, t]
            return (m.gamma*(1 - t**2)*m.drhodt[p, vol+1, t]*V/(N-1)
                    == SEC_PER_HOUR*(w*scale_w - w_next*scale_w - cons*scale_w))

        m.PIPE_mass_balance = pyo.Constraint(
            m.Pipes_VolCenterC, m.Times,
            rule=_pipe_mass_balance_infhorizon_hours)

    ######################################################
    # Special for Gaussian quadrature
    ######################################################

    def _tail_css_index(self, tau):
        """Map a transformed tail point to the periodic OCSS reference."""
        t_bar = self.Horizon
        gamma = pyo.value(self.m_controller_inf.gamma)
        physical_time = t_bar + np.arctanh(float(tau))/gamma
        return (
            int(round(physical_time % self.Cycle_length))
            % self.Cycle_length
        )

    def _build_tail_radau_weights(self):
        """Return composite Radau weights, excluding only the singular tau=1."""
        inf = self.m_controller_inf
        finite_elements = list(inf.Times.get_finite_elements())
        all_points = sorted(inf.Times)
        weights = {}
        for element in range(len(finite_elements) - 1):
            left = finite_elements[element]
            right = finite_elements[element + 1]
            radau_points = [tau for tau in all_points if left < tau <= right]
            if len(radau_points) != self.NCP_infinite_horizon:
                raise RuntimeError(
                    f"Expected {self.NCP_infinite_horizon} Radau points "
                    f"in [{left}, {right}], got {len(radau_points)}")

            # Integrate the Lagrange basis on [0, 1].  Solving the moment
            # equations makes the weights match Pyomo's actual Radau nodes.
            local_points = np.array(
                [(float(tau) - float(left))/(float(right) - float(left))
                 for tau in radau_points])
            moment_matrix = np.vstack([
                local_points**degree
                for degree in range(self.NCP_infinite_horizon)
            ])
            moments = np.array([
                1.0/(degree + 1)
                for degree in range(self.NCP_infinite_horizon)
            ])
            reference_weights = np.linalg.solve(moment_matrix, moments)
            for tau, omega in zip(radau_points, reference_weights):
                if abs(float(tau) - 1.0) <= 1.0e-12:
                    continue
                weights[tau] = (right - left)*float(omega)
        return weights

    # ============================================================================
    # Infinite-horizon controller
    # ============================================================================
    def make_infinite_horizon_controller(self):
        # Load network and input data
        networkData, inputData = import_data_from_excel(self.network_data_path, self.input_data_path)
        
        # Load options file
        with open(self.options_data_path, 'r') as file:
            Options = json.load(file)
        Options['dynamic']=True
        
        self.m_controller_inf = buildNonLinearModel(networkData, inputData, Options, infinite_horizon=True)

        self._rebuild_infinite_mass_balance()

        inf = self.m_controller_inf
        NFE = self.NFE_infinite_horizon
        NCP = self.NCP_infinite_horizon
        discretizer = pyo.TransformationFactory('dae.collocation')
        discretizer.apply_to(
            inf,
            wrt=inf.Times,
            nfe=NFE,
            ncp=NCP,
            scheme='LAGRANGE-RADAU',
        )

        # Reduce collocation points in the infinite horizon for the controls
        discretizer.reduce_collocation_points(self.m_controller_inf,
                                              var = self.m_controller_inf.compressor_P,
                                              ncp = 1,
                                              contset = self.m_controller_inf.Times)
        #Initialize 
        self.m_controller_inf = init_network_default(self.m_controller_inf, p_default = 55e5)
        self.m_controller_inf.wCons.fix(0)
        # Gamma
        self.m_controller_inf.gamma = self.Gamma

        # Demands need to be converted to the tau domain
        sink_node_set = [
            s for s in self.m_controller_inf.Nodes
            if str(s).startswith(("sink", "exit"))
        ]
        self.m_controller_inf.sink_node_set = pyo.Set(initialize = sink_node_set)
        for tau in self.m_controller_inf.Times:
            if tau == 1:
                break
            t_bar = self.Horizon 
            gamma = pyo.value(self.m_controller_inf.gamma)
            t = t_bar + 1/gamma*np.arctanh(tau)
    
            sin_func = np.sin(2*np.pi/self.Cycle_length*t)
            
            for s in self.m_controller_inf.sink_node_set:
                node_nominal_demand = self.nominal_demand[s]
                self.m_controller_inf.wCons[s, 0, tau].fix(
                    node_nominal_demand
                    + node_nominal_demand/20*sin_func)

        # Deactivate differential equation at the terminal point
        # because it is a steady state equation
        # but the endpoint constraint is cyclic ss
        tau_f = self.m_controller_inf.Times.last()

        self.m_controller_inf.PIPE_mass_balance[:,:,tau_f].deactivate()
        
        # GasLib-40: source_1/source_2 specify pressure, while source_3
        # specifies flow. Keep each source's boundary-condition type.
        source_1_pressure = self.dynamic_source_pressure_bar
        source_2_pressure = self.dynamic_source_pressure_bar
        source_3_flow = pyo.value(
            self.m_controller_inf.wSource["source_3", 0])
        for tau in self.m_controller_inf.Times:
            self.m_controller_inf.pSource[
                "source_1", tau].fix(source_1_pressure)
            self.m_controller_inf.pSource[
                "source_2", tau].fix(source_2_pressure)
            self.m_controller_inf.wSource[
                "source_3", tau].fix(source_3_flow)

        # Preserve the efficiency value belonging to each compressor.
        for station in self.m_controller_inf.Stations:
            eta = pyo.value(
                self.m_controller_inf.compressor_eta[station, 0])
            self.m_controller_inf.compressor_eta[station, :].fix(eta)

        ##################################################################
        # Composite Radau quadrature; omit the singular final point tau=1.
        gamma = self.m_controller_inf.gamma
        self._tail_radau_weights = self._build_tail_radau_weights()
        self.m_controller_inf.stage_cost = 1/1000*sum(
            self.m_controller_inf.compressor_P[s, tau]
            * SEC_PER_HOUR/(gamma*(1 - tau**2))
            * weight
            for s in self.m_controller_inf.Stations
            for tau, weight in self._tail_radau_weights.items())

        # Deactivate objective function
        self.m_controller_inf.ObjFun.deactivate()

    def initialize_infinite_tail_from_finite_endpoint(self):
        """Warm-start the infinite tail from the finite endpoint."""
        finite = self.m_controller_finite
        infinite = self.m_controller_inf
        tf = finite.Times.last()

        shared_components = (
            "compressor_P",
            "compressor_beta",
            "node_p",
            "interm_w",
            "interm_p",
            "wSource",
            "pSource",
            "pipe_rho",
            "inlet_w",
            "outlet_w",
            "u",
            "u2",
        )

        initialized = 0
        for component_name in shared_components:
            finite_component = getattr(finite, component_name)
            infinite_component = getattr(infinite, component_name)
            for tail_index in infinite_component:
                index = (
                    tail_index
                    if isinstance(tail_index, tuple)
                    else (tail_index,)
                )
                finite_index = index[:-1] + (tf,)
                if finite_index not in finite_component:
                    continue
                tail_value = infinite_component[tail_index]
                if tail_value.fixed:
                    continue
                tail_value.set_value(
                    pyo.value(finite_component[finite_index])
                )
                initialized += 1

        # A constant endpoint trajectory corresponds to a zero initial
        # density derivative. IPOPT remains free to change these values.
        if hasattr(infinite, "drhodt"):
            for derivative_value in infinite.drhodt.values():
                if not derivative_value.fixed:
                    derivative_value.set_value(0.0)

        print(
            f"Infinite tail initialized from finite endpoint t={tf}: "
            f"{initialized} values",
            flush=True,
        )

    # ============================================================================
    # Finite and infinite controller assembly
    # ============================================================================
    def make_actual_controller_model(self):
        self.m_controller = pyo.ConcreteModel()
        self.m_controller.finite = self.m_controller_finite
        self.m_controller.infinite = self.m_controller_inf

        # Connect all variables between finite and infinte horizons
        #### Link variables ####
        tf_finite = self.m_controller.finite.Times.last()
        t0_infinite = self.m_controller.infinite.Times.first()

        ##==== Node vars =====##
        # Link node pressures
        def _link_node_pressures(m, n):
            return m.finite.node_p[n, tf_finite] == m.infinite.node_p[n, t0_infinite]
        #self.m_controller.link_node_p = pyo.Constraint(self.m_controller.finite.Nodes, rule = _link_node_pressures)
        
        # Link wSource - source flows
        def _link_source_flow(m, n):
            return m.finite.wSource[n, tf_finite] == m.infinite.wSource[n, t0_infinite]
        #self.m_controller.link_source_flow = pyo.Constraint(self.m_controller.finite.NodesSources, rule = _link_source_flow)
        
        ## =====  Arc vars ==== ##
        # Link inlet_w - Inlet flows in pipe arcs
        def _link_arc_inlet_flows(m, a):
            return m.finite.inlet_w[a, tf_finite] == m.infinite.inlet_w[a, t0_infinite]
        #self.m_controller.link_arc_inlet_flows = pyo.Constraint(self.m_controller.finite.Arcs, rule = _link_arc_inlet_flows)
        
        # Link outlet_w - Outlet flow in pipe arcs
        def _link_arc_outlet_flows(m, a):
            return m.finite.outlet_w[a, tf_finite] == m.infinite.outlet_w[a, t0_infinite]
        #self.m_controller.link_arc_outlet_flows = pyo.Constraint(self.m_controller.finite.Arcs, rule = _link_arc_outlet_flows)
        
        ##==== Station vars =====##
        def _link_compressor_P(m, s):
            return m.finite.compressor_P[s, tf_finite] == m.infinite.compressor_P[s, t0_infinite]
        self.m_controller.link_compressor_P = pyo.Constraint(self.m_controller.finite.Stations, rule = _link_compressor_P)
        
        def _link_compressor_Beta(m, s):
            return m.finite.compressor_beta[s, tf_finite] == m.infinite.compressor_beta[s, t0_infinite]
        #self.m_controller.link_compressor_beta = pyo.Constraint(self.m_controller.finite.Stations, rule = _link_compressor_Beta)
        
        ## ==== Pipe vars ==== ##
        # Link interm_w - Intermediate pipe flows
        def _link_interm_pipe_flows(m, p1, p2):
            return m.finite.interm_w[p1, p2, tf_finite] == m.infinite.interm_w[p1, p2, t0_infinite]
        #self.m_controller.link_interm_pipe_flows = pyo.Constraint(self.m_controller.finite.Pipes_VolExtrC_interm, rule = _link_interm_pipe_flows)
        
        # Link interm_p - Intermediate pipe pressures
        def _link_interm_pipe_pressures(m, p1, p2):
            return m.finite.interm_p[p1, p2, tf_finite] == m.infinite.interm_p[p1, p2, t0_infinite]
        self.m_controller.link_interm_pipe_pressures = pyo.Constraint(self.m_controller.finite.Pipes_VolExtrR_interm, rule = _link_interm_pipe_pressures)
        
        # Link u - Velocity
        def _link_velocity(m, p1, p2):
            return m.finite.u[p1, p2, tf_finite] == m.infinite.u[p1, p2, t0_infinite]
        #self.m_controller.link_u = pyo.Constraint(self.m_controller.finite.Pipes_VolExtrC, rule = _link_velocity)
       
        #Link u**2 - Velocity square
        def _link_velocity_square(m, p1, p2):
            return m.finite.u2[p1, p2, tf_finite] == m.infinite.u2[p1, p2, t0_infinite]
        #self.m_controller.link_u2 = pyo.Constraint(self.m_controller.finite.Pipes_VolExtrC, rule = _link_velocity_square)
        
        # Link pipe_rho - density in pipes
        def _link_pipe_density(m, p1, p2):
            return m.finite.pipe_rho[p1, p2, tf_finite] == m.infinite.pipe_rho[p1, p2, t0_infinite]
        #self.m_controller.link_pipe_density = pyo.Constraint(self.m_controller.finite.Pipes_VolExtrR, rule =_link_pipe_density)

        #### ========= Optional soft terminal constraints ========== ####
        if self.terminal:
            inf = self.m_controller.infinite
            inf.terminal_p_slack_pos = pyo.Var(
                inf.Pipes_VolExtrR_interm,
                initialize=0, domain=pyo.NonNegativeReals)
            inf.terminal_p_slack_neg = pyo.Var(
                inf.Pipes_VolExtrR_interm,
                initialize=0, domain=pyo.NonNegativeReals)
            inf.terminal_P_slack_pos = pyo.Var(
                inf.Stations, initialize=0, domain=pyo.NonNegativeReals)
            inf.terminal_P_slack_neg = pyo.Var(
                inf.Stations, initialize=0, domain=pyo.NonNegativeReals)

            def _terminal_pressure(m, p, vol):
                return (m.interm_p[p, vol, 1]
                        == self.m_controller.finite.interm_p_ocss[p, vol, 0]
                        + m.terminal_p_slack_pos[p, vol]
                        - m.terminal_p_slack_neg[p, vol])
            inf.terminal_pressure = pyo.Constraint(
                inf.Pipes_VolExtrR_interm, rule=_terminal_pressure)

            def _terminal_comp_power_css(m, s):
                return (m.compressor_P[s, 1]
                        == self.m_controller.finite.compressor_P_ocss[s, 0]
                        + m.terminal_P_slack_pos[s]
                        - m.terminal_P_slack_neg[s])
            inf.terminal_compressor_power_css = pyo.Constraint(
                inf.Stations, rule=_terminal_comp_power_css)

            inf.terminal_cost = (
                sum(inf.terminal_p_slack_pos[p, vol]
                    + inf.terminal_p_slack_neg[p, vol]
                    for p, vol in inf.Pipes_VolExtrR_interm)
                + sum(inf.terminal_P_slack_pos[s]
                      + inf.terminal_P_slack_neg[s]
                      for s in inf.Stations)
            )
        
        for s in self.m_controller.infinite.sink_node_set:
            self.m_controller.infinite.wCons[s,:,1].fix(pyo.value(self.m_controller.finite.wCons[s, 0, self.Horizon]))

    # ============================================================================
    # Pressure, compressor-power, and source-3 pressure tracking
    # ============================================================================
    def make_tracking_cost(self):
        m = self.m_controller
        fin = m.finite
        inf = m.infinite

        m.tracking_p_finite = self.coeff_p*sum(
            (fin.interm_p[p, vol, t]
             - fin.interm_p_ocss[p, vol, t])**2
            for p, vol in fin.Pipes_VolExtrR_interm
            for t in fin.Times if t != fin.Times.last()
        )
        m.tracking_P_finite = self.coeff_P*sum(
            (fin.compressor_P[s, t]
             - fin.compressor_P_ocss[s, t])**2
            for s in fin.Stations
            for t in fin.Times if t != fin.Times.last()
        )
        m.tracking_pSource3_finite = self.coeff_pSource3*sum(
            (fin.pSource["source_3", t]
             - fin.pSource_ocss["source_3", t])**2
            for t in fin.Times if t != fin.Times.last()
        )

        gamma = inf.gamma
        m.tracking_p_infinite = self.coeff_p*sum(
            (inf.interm_p[p, vol, tau]
             - fin.interm_p_ocss[
                 p, vol, self._tail_css_index(tau)])**2
            / (gamma*(1 - tau**2))*weight
            for p, vol in inf.Pipes_VolExtrR_interm
            for tau, weight in self._tail_radau_weights.items()
        )
        m.tracking_P_infinite = self.coeff_P*sum(
            (inf.compressor_P[s, tau]
             - fin.compressor_P_ocss[
                 s, self._tail_css_index(tau)])**2
            / (gamma*(1 - tau**2))*weight
            for s in inf.Stations
            for tau, weight in self._tail_radau_weights.items()
        )
        m.tracking_pSource3_infinite = self.coeff_pSource3*sum(
            (inf.pSource["source_3", tau]
             - fin.pSource_ocss[
                 "source_3", self._tail_css_index(tau)])**2
            / (gamma*(1 - tau**2))*weight
            for tau, weight in self._tail_radau_weights.items()
        )

        m.tracking_cost = (
            m.tracking_p_finite + m.tracking_p_infinite
            + m.tracking_P_finite + m.tracking_P_infinite
            + m.tracking_pSource3_finite
            + m.tracking_pSource3_infinite
        )

    # ============================================================================
    # Objective function
    # ============================================================================
    def make_objective_function(self):
        expr = self.m_controller.tracking_cost
        if self.terminal:
            expr += (self.terminal_coeff
                     * self.m_controller.infinite.terminal_cost)
        self.m_controller.obj = pyo.Objective(expr = expr)

    # ============================================================================
    # NMPC execution
    # ============================================================================
    def run_nmpc(self):
        _total_start = time.time()
        timer = HierarchicalTimer()
        timer.start("Initialization")

        # Build the finite-horizon controller and the one-sample plant model.
        ########################### 1st, 2nd, 3rd, 4th solves in make_controller_plant.py ###########################
        self.make_controller_and_plant_model_finite()

        # Build and discretize the infinite-horizon tail model.
        self.make_infinite_horizon_controller()

        # Initialize the tail before assembling and solving MPC 0.
        self.initialize_infinite_tail_from_finite_endpoint()

        # Assemble the finite and infinite models into one optimization model.
        self.make_actual_controller_model()

        self.make_tracking_cost()

        # Create the combined finite, infinite-tail, and terminal objective.
        self.make_objective_function()

        # Local alias for the finite part used in the receding-horizon loop.
        model = self.m_controller.finite

        # The controller data source is also the finite part in this formulation.
        input_data_model = self.m_controller.finite

        t0 = model.Times.first()

        # Non initial plant time
        non_initial_plant_time = [t for t in self.m_plant.Times if t != self.m_plant.Times.first()]

        # For record
        self.sim_data = self.plant_interface.get_data_at_time([0])
        self.objective_terms = {}
        self.iteration_times = {}

        timer.stop("Initialization")

        for step, i in enumerate(
                np.arange(0, self.Simulation_period, self.Sampling_time)):
            timer.start(f"iteration_{step}")
            t_base = i

            print(f"\n================ MPC {step} | t = {float(t_base):.1f} h ================",flush=True)

            if self.unfix_P_at_t0:
                model.compressor_P[:, t0].unfix()

            # Solve controller model
            ########################### Controller solve ###########################
            #============= Controller solve and reporting ===============
            controller_solve_start = time.time()
            timer.start("Solve_controller")
            res = self.ipopt.solve(self.m_controller, tee= True)
            timer.stop("Solve_controller")
            controller_solve_time = time.time() - controller_solve_start

            pyo.assert_optimal_termination(res)

            self.objective_terms[t_base] = pyo.value(self.m_controller.obj)

            tracking_p = pyo.value(
                self.m_controller.tracking_p_finite
                + self.m_controller.tracking_p_infinite)
            tracking_P = pyo.value(
                self.m_controller.tracking_P_finite
                + self.m_controller.tracking_P_infinite)
            tracking_pSource3 = pyo.value(
                self.m_controller.tracking_pSource3_finite
                + self.m_controller.tracking_pSource3_infinite)
            terminal_penalty = (
                self.terminal_coeff*pyo.value(
                    self.m_controller.infinite.terminal_cost)
                if self.terminal else 0.0
            )
            print(
                f"  Objective: tracking_p={tracking_p:.6e}, "
                f"tracking_P={tracking_P:.6e}, "
                f"tracking_pSource3={tracking_pSource3:.6e}, "
                f"terminal={terminal_penalty:.6e}, "
                f"total={self.objective_terms[t_base]:.6e}",
                flush=True,
            )
            print(f"  Controller solve time: {controller_solve_time:.2f} s", flush=True)
            #============= Controller solve and reporting ===============

            # Extract t1 controls from controller 
            ts_data = self.controller_interface.get_data_at_time(self.Sampling_time)
            input_data = ts_data.extract_variables(self.plant_fixed_variables, context = input_data_model)

            # Simulate the plant
            self.plant_interface.load_data(input_data, time_points=non_initial_plant_time)
            
            for s in self.m_plant.sink_node_set:
                for t in self.m_plant.Times:
                    self.m_plant.actual_demand[s,t] = pyo.value(input_data_model.actual_demand[s,t])
                    self.m_plant.wCons[s, 0, t].fix(pyo.value(self.m_plant.actual_demand[s,t]))
            assert degrees_of_freedom(self.m_plant) == 0
            ########################### Plant solve ###########################
            timer.start("Solve_plant")
            res = self.ipopt.solve(self.m_plant, tee=True)
            timer.stop("Solve_plant")
            pyo.assert_optimal_termination(res)

            # Extract data from simulated model
            m_data = self.plant_interface.get_data_at_time(non_initial_plant_time)
            m_data.shift_time_points(t_base - self.m_plant.Times.first())
            self.sim_data.concatenate(m_data)
            
            # Extract data from the last time point in the plant
            tf_data = self.plant_interface.get_data_at_time(self.m_plant.Times.last())
         
            # Reinitialize plant model
            self.plant_interface.load_data(tf_data)
        
            # Reinitialize controller 
            self.controller_interface.shift_values_by_time(self.Sampling_time)
            self.controller_interface.load_data(tf_data, time_points=t0)

            # Note - Here terminal constraint is automatically moved 
            # because the ocss parameters are declared as fixed variables
            # They are automatically moved when we shift values by time 

            # Move the parameters by the sampling time 
            self.shift_time_indexed_parameters_finite(
                float(t_base) + self.Sampling_time)
            
            # Update ocss at the last point to be equal to the first point
            N = self.NumCycles
            K = int(self.Horizon/N)

            [model.compressor_P_ocss[s, N*K].fix(model.compressor_P_ocss[s, (N-1)*K]) for s in model.Stations]
            [model.interm_p_ocss[p, vol, N*K].fix(model.interm_p_ocss[p, vol, (N-1)*K]) for p, vol in model.Pipes_VolExtrR_interm]
            [model.compressor_beta_ocss[s, N*K].fix(model.compressor_beta_ocss[s, (N-1)*K]) for s in model.Stations]
            model.pSource_ocss["source_3", N*K].fix(
                model.pSource_ocss["source_3", (N-1)*K])

            timer.stop(f"iteration_{step}")
            self.iteration_times[step] = timer.get_total_time(
                f"iteration_{step}")

        self.total_runtime = time.time() - _total_start
        return self.m_plant, self.m_controller, self.sim_data

    # ============================================================================
    # Results reporting and saving
    # ============================================================================
    def save_results(self):
        all_nodes_p = []
        for n in self.m_plant.Nodes:
            if str(n).startswith(("sink", "exit")):
                all_nodes_p.append(self.m_plant.node_p[n, :])

        sheets_keys_dict = {
            "compressor power": [
                self.m_plant.compressor_P[s, :]
                for s in self.m_plant.Stations
            ],
            "compressor beta": [
                self.m_plant.compressor_beta[s, :]
                for s in self.m_plant.Stations
            ],
            "wCons": [
                self.m_plant.wCons[s, 0, :]
                for s in self.m_plant.Nodes
                if str(s).startswith(("sink", "exit"))
            ],
            "node pressure": all_nodes_p,
            "interm_w": [
                self.m_plant.interm_w[p, vol, :]
                for p, vol in self.m_plant.Pipes_VolExtrC_interm
            ],
            "interm_p": [
                self.m_plant.interm_p[p, vol, :]
                for p, vol in self.m_plant.Pipes_VolExtrR_interm
            ],
            "wSource": [
                self.m_plant.wSource[s, :]
                for s in self.m_plant.NodesSources
            ],
            "pSource": [
                self.m_plant.pSource[s, :]
                for s in self.m_plant.NodesSources
            ],
        }

        write_data_to_excel(
            self.sim_data,
            self.m_plant,
            sheets_keys_dict,
            self.results_path,
        )

        terms = getattr(self, "objective_terms", None)
        if terms:
            objective_df = pd.Series(terms, name="obj").to_frame()
            objective_df = objective_df.reindex(sorted(objective_df.index))
            objective_df.index.name = "t [h]"
            with pd.ExcelWriter(
                    self.results_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="replace") as writer:
                objective_df.to_excel(writer, sheet_name="objective")

        iteration_times = getattr(self, "iteration_times", None)
        if iteration_times:
            runtime_df = pd.DataFrame({
                "iteration": list(iteration_times.keys()),
                "iteration_time_s": list(iteration_times.values()),
            })
            runtime_df.loc[len(runtime_df)] = [
                "TOTAL",
                getattr(self, "total_runtime", float("nan")),
            ]
            with pd.ExcelWriter(
                    self.results_path,
                    engine="openpyxl",
                    mode="a",
                    if_sheet_exists="replace") as writer:
                runtime_df.to_excel(
                    writer, sheet_name="runtime", index=False)

        print(f"Results saved to: {self.results_path}")


# ============================================================================
# Main execution
# ============================================================================
if __name__ == "__main__":
    RESULTS_PATH = str(BASE_PATH / "Inf_tracking_e-1.xlsx")
    LOG_PATH = str(BASE_PATH / "Inf_tracking_e-1.txt")
    TERMINAL = False

    class _Tee:
        def __init__(self, log_path):
            import sys
            self._terminal = sys.stdout
            self._log = open(log_path, "w", buffering=1)
            sys.stdout = self

        def write(self, message):
            self._terminal.write(message)
            self._log.write(message)

        def flush(self):
            self._terminal.flush()
            self._log.flush()

        def close(self):
            import sys
            sys.stdout = self._terminal
            self._log.close()

    tee = _Tee(LOG_PATH)
    try:
        gasnet = ControlGasNet()
        gasnet.terminal = TERMINAL
        gasnet.results_path = RESULTS_PATH
        gasnet.log_path = LOG_PATH
        print("terminal =", gasnet.terminal)
        print("coeff_p =", gasnet.coeff_p)
        print("coeff_P =", gasnet.coeff_P)
        print("coeff_pSource3 =", gasnet.coeff_pSource3)
        print("terminal_coeff =", gasnet.terminal_coeff)
        gasnet.run_nmpc()
        gasnet.save_results()

        total_power = sum(
            sum(gasnet.sim_data.get_data_from_key(
                f"compressor_P['{station}',*]"))
            for station in gasnet.m_plant.Stations
        )
        total_beta = sum(
            sum(gasnet.sim_data.get_data_from_key(
                f"compressor_beta['{station}',*]"))
            for station in gasnet.m_plant.Stations
        )
        total_mpc_time = sum(gasnet.iteration_times.values())

        print("Total power = ", total_power)
        print("Total beta  = ", total_beta)
        print("Total MPC time = ", total_mpc_time)
        print(f"Log saved to: {LOG_PATH}")
    finally:
        tee.close()
        


                

