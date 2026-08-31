import pyomo.environ as pyo
import pyomo.dae as dae
import numpy as np
import math

#==============================================================================
""" SETS """
#==============================================================================

''' Time '''
def TIME_sets(m, Opt):
    T0 = Opt['T0'] # start
    T = Opt['T'] # ends
    dt = Opt['dt']
    times = np.arange(T0, T+dt/3600, dt/3600) # nmpc library works only with continuous set, np.arange(start, stop, step)
    m.Times = dae.ContinuousSet(initialize=times)

    return m # update m

''' Network '''
def NODE_sets(m, networkData):
    Nodes = networkData['Nodes']
    Arcs = networkData['Arcs']

    # Data
    for n in Nodes.keys():
        Nodes[n]['arcsOUT'] = []
        Nodes[n]['arcsIN'] = []
    for arc in Arcs.keys():
        n_in = Arcs[arc]['nodeIN']
        n_out = Arcs[arc]['nodeOUT']
        Nodes[n_in]['arcsOUT'].append(arc)
        Nodes[n_out]['arcsIN'].append(arc)
    Nodes_ArcsIN = {n: Nodes[n]['arcsIN'] for n in Nodes.keys()}
    Nodes_ArcsOUT = {n: Nodes[n]['arcsOUT'] for n in Nodes.keys()}
    NodesSources = [n for n in Nodes.keys() if str(Nodes[n]['source']) != 'nan']
    # All nodes
    m.Nodes = pyo.Set(initialize=Nodes.keys())
    # Supply nodes
    m.NodesSources = pyo.Set(initialize=NodesSources)
    # Multi-layer sets
    m.Nodes_ArcsIN = pyo.Set(m.Nodes, initialize=Nodes_ArcsIN)
    m.Nodes_ArcsOUT = pyo.Set(m.Nodes, initialize=Nodes_ArcsOUT)
    return m

def ARC_sets(m, networkData):
    Arcs = networkData['Arcs']
    m.Arcs = pyo.Set(initialize=list(Arcs.keys()))
    # NODES IN/OUT from each ARC (not actually a pyomo Set)
    m.Arcs_NodeIN = {arc: Arcs[arc]['nodeIN'] for arc in Arcs.keys()}
    m.Arcs_NodeOUT = {arc: Arcs[arc]['nodeOUT'] for arc in Arcs.keys()}

    return m

''' Valves '''
def VALVE_sets(m, networkData):
    m.Valves = pyo.Set(initialize=networkData['Valves'].keys())

    return m

''' Stations / Machines '''
def STATION_set(m, networkData):
    m.Stations = pyo.Set(initialize=networkData['Stations'].keys())

    return m

''' Pipes '''
def PIPE_sets(m, networkData):
    Pipes = networkData['Pipes']

    # Sets: pipes
    m.Pipes = pyo.Set(initialize = Pipes.keys())

    # Sets: finite volumn pipes
    Pipes_VolExtrR = [] # pressure node, 12345, constraint: momentum balance; variables: density
    Pipes_VolCenterC = [] # control volumn (CV), 123, constraint: mass balance
    Pipes_VolExtrC = [] # flowrate node, 1234, variables: speed u, squared speed u2
    Pipes_VolExtrR_interm = [] # pressure node, 234, variables: pressure in the middle of the pipe
    Pipes_VolExtrC_interm = [] # flowrate node, 23, variables: mass flow rate in the middle of the pipe
    for p in Pipes.keys():
        N = Pipes[p]['Nvol']
        for index in range(1, N+2): # N = 4, index = 1, 2, 3, 4, 5
            Pipes_VolExtrR.append((p, index)) # tuple()
            if index != 1 and index != N+1:
                Pipes_VolExtrR_interm.append((p, index))
            if index != N+1:
                Pipes_VolExtrC.append((p, index))
                if index != N:
                    Pipes_VolCenterC.append((p, index))
                    if index !=1:
                        Pipes_VolExtrC_interm.append((p,index))

    m.Pipes_VolExtrR = pyo.Set(initialize=Pipes_VolExtrR)
    m.Pipes_VolExtrC = pyo.Set(initialize = Pipes_VolExtrC)
    m.Pipes_VolExtrR_interm = pyo.Set(initialize = Pipes_VolExtrR_interm)
    m.Pipes_VolExtrC_interm = pyo.Set(initialize = Pipes_VolExtrC_interm)
    m.Pipes_VolCenterC = pyo.Set(initialize = Pipes_VolCenterC)

    return m

#==============================================================================
""" PARAMETERS """
#==============================================================================
''' Gas '''
def PIPE_gas_params(m, networkData):
    Pipes = networkData['Pipes']

    # Gas properties
    Gas = {}
    for arc in m.Pipes.data():
        for par in ['MMgas', 'Tgas', 'Z']:
            Gas[(arc, par)] = Pipes[arc][par]

    m.Gas = pyo.Param(Gas.keys(), initialize = Gas, within = pyo.Reals)

    return m

''' Geometry '''
def PIPE_geom_params(m, networkData):
    Pipes = networkData['Pipes']
    # number of finite volumn per pipe
    Nvol = {p: Pipes[p]['Nvol'] for p in m.Pipes}
    m.Nvol = pyo.Param(m.Pipes, initialize=Nvol, default=2, within=pyo.Integers)
    # area of the pipe
    A = {p: Pipes[p]['A'] for p in m.Pipes}
    m.Area = pyo.Param(m.Pipes, initialize=A, within=pyo.NonNegativeReals)
    # length of the pipe
    L = {p: Pipes[p]['length'] for p in m.Pipes}
    m.Length = pyo.Param(m.Pipes, initialize=L, within=pyo.NonNegativeReals)
    # diameter of the pipe
    D = {p: Pipes[p]['diameter'] for p in m.Pipes}
    m.Diam = pyo.Param(m.Pipes, initialize=D, within=pyo.NonNegativeReals)
    # roughness of the pipe
    rough = {p: Pipes[p]['roughness'] for p in m.Pipes}
    m.Roughness = pyo.Param(m.Pipes, initialize=rough, within=pyo.NonNegativeReals)
    # wet parameter of the pipe (omega = pi * D)
    omega = {p: Pipes[p]['omega'] for p in m.Pipes}
    m.Omega = pyo.Param(m.Pipes, initialize=omega, within=pyo.NonNegativeReals)

    return m

''' Flow reversal '''
# used for u^2 = u * sqrt(u^2 + epsilon)
def PIPE_smoothing_params(m, eps):
    m.eps = pyo.Param(initialize=eps, default=1e-4, mutable=True) # mutable = can be changed during calculation

    return m


#==============================================================================
""" VARIABLES """
#==============================================================================

''' Network '''
def NODE_vars(m):
    # All nodes
    m.node_p = pyo.Var(m.Nodes, m.Times, within=pyo.NonNegativeReals)
    # Remark: node demand (wCons) is defined together with pipes demand
    # Supply nodes
    m.pSource = pyo.Var(m.NodesSources, m.Times, within=pyo.NonNegativeReals)
    m.wSource = pyo.Var(m.NodesSources, m.Times, within = pyo.NonNegativeReals)

    return m

def ARC_vars(m):
    # Arcs
    m.inlet_w = pyo.Var(m.Arcs, m.Times, within = pyo.Reals)
    m.outlet_w = pyo.Var(m.Arcs, m.Times, within = pyo.Reals)

    return m

''' Station '''
def STATIONS_vars(m):
    m.compressor_P = pyo.Var(m.Stations, m.Times, within=pyo.NonNegativeReals)
    m.compressor_beta = pyo.Var(m.Stations, m.Times, bounds=(1.05, 2), within=pyo.NonNegativeReals)
    m.compressor_eta = pyo.Var(m.Stations, m.Times, bounds=(0, 1), within=pyo.NonNegativeReals)
    m.compressor_eta.fix(0.8)

    return m

''' Pipes '''
def PIPE_vars(m, scale, networkData):
    # finite volumes --> variables in the middle of the pipes (at boundaries, pressure = node_p, mass flow = inlet_w/outlet_w)
    m.interm_w = pyo.Var(
        m.Pipes_VolExtrC_interm, m.Times, within=pyo.Reals
    )
    # pressure bounds 
    p_min = min(networkData['Nodes'][key]['pMin'] for key in networkData['Nodes'].keys())
    p_max = max(networkData['Nodes'][key]['pMax'] for key in networkData['Nodes'].keys())
  
    m.interm_p = pyo.Var(
        m.Pipes_VolExtrR_interm, m.Times, bounds=(p_min/scale['p'], p_max/scale['p']), within=pyo.NonNegativeReals
    )
    # friction
    m.u2 = pyo.Var(m.Pipes_VolExtrC, m.Times, within=pyo.Reals)
    m.u = pyo.Var(m.Pipes_VolExtrC, m.Times, within = pyo.Reals)
    # density
    temperature = min(networkData['Pipes'][key]['Tgas'] for key in networkData['Pipes'].keys())
    MMgas = min(networkData['Pipes'][key]['MMgas'] for key in networkData['Pipes'].keys())

    rho_min = p_min * MMgas / (8314*temperature)
    rho_max = p_max * MMgas / (8314*temperature)

    m.pipe_rho = pyo.Var(m.Pipes_VolExtrR, m.Times, bounds=(rho_min, rho_max), within=pyo.NonNegativeReals)
    # consumption variables --> nodes and pipes
    wcons_keys = list(m.Pipes_VolCenterC.data()) + [(n, 0) for n in m.Nodes.data() if n not in m.NodesSources.data()]
    m.wCons = pyo.Var(wcons_keys, m.Times, within=pyo.Reals)

    return m

''' Dual variables '''
def DUALS_ipopt_vars(m):
    # Ipopt bound multipliers (import)
    m.ipopt_zL_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    m.ipopt_zU_out = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    # Ipopt bound multipliers (EXPORT)
    m.ipopt_zL_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.ipopt_zU_in = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    # Ipopt constraint dual (IMPORT/EXPORT)
    m.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT_EXPORT)

    return m

# ========================================================================
''' Functions: finite volume method - pipes'''
# ========================================================================

''' Evaluate gas density in pipes finite volumes '''
def Pipe_GasDensity(
        m, scale, p, vol, t, N, 
        calc = True):
    # Calculate gas density as a constant, or return as a variable to be solve by pyomo
    if calc:
        # Gas parameters 
        Tgas = m.Gas[p,"Tgas"]
        MM = m.Gas[p,"MMgas"] # Molecular mass
        Zavrg = m.Gas[p,"Z"]
        # Equation
        press = Pipe_Pressure(m, p, vol, t, N)*scale["p"]  
        rho = press / (8314 / MM * Tgas * Zavrg)
        return rho
    # Pyomo variable
    else:
        rho = m.pipe_rho[p, vol, t]
        return rho
    
''' Evaluate pressure in pipes finite volumes '''
def Pipe_Pressure(
        m, p, vol, t, N):
    # Find pressure of Finite Volume of pipe
    start = 1
    end = N + 1    
    # Pressures in the middle of the pipes
    if vol != start and vol != end:
        press = m.interm_p[p, vol, t] 
    # Pressure at the outlet
    elif vol == end:
        n_out = m.Arcs_NodeOUT[p]
        press = m.node_p[n_out, t]
    # Pressure at the inlet
    elif vol == start:
        n_in = m.Arcs_NodeIN[p]
        press = m.node_p[n_in, t] 
    return press    

''' Evaluate mass flow rate in pipes finite volumes '''
def Pipe_MassFlow(
        m, p, vol, t, N):
    # Find mass flow rate of Finite Volume of pipe
    start = 1
    end = N    
    # Flow at the inlet   
    if vol == start:
        w = m.inlet_w[p, t] 
    # Flow at the outlet
    elif vol == end:
        w = m.outlet_w[p, t] 
    # Flow in the middle of the pipes 
    else:
        w = m.interm_w[p, vol, t] 
    return w

''' Evaluate dmdt 
    Remark: density is evaluated on the staggered grid (next volume)
    That is to use "v+1" to evaluate density for "v" '''

def Pipe_DerMass(
        m, p, vol, t, N, V, scale, dt
        ):  
    t_prev = m.Times.prev(t) # find previous time step, eg: if t = 2, then t_prev = 1
    # Density at the staggered grid (next volume)
    rho = Pipe_GasDensity(m, scale, p, vol+1, t, N, calc = False) # return density as a variable for further derivative evaluation
    # Density at previous time step
    rho_prev = Pipe_GasDensity(m, scale, p, vol+1, t_prev, N, calc = False) # return density as a constant for calculation
    # Derivative term in mass balance
    dMdt = (rho - rho_prev) * V / (N-1) / dt
    return dMdt

#==============================================================================
""" OBJECTIVE FUNCTION """
#==============================================================================

def OBJ_compressor_power(m, Opt):
    dt = Opt['dt']
    def ObjFun(m):
        Obj = sum(m.compressor_P[s, t]*dt for s in m.Stations for t in m.Times if t != m.Times.last())
        return Obj / 1000
    m.ObjFun = pyo.Objective(rule = ObjFun, sense = 1) # sense = 1 <=> minimize, sense = -1 <=> maximize
    return m


#==============================================================================
""" CONSTRAINTS - NODE """
#==============================================================================

''' Simple nodes '''
def NODE_constr(m, scale, networkData):
    Nodes = networkData["Nodes"]

    # Pressure
    for n in m.Nodes:
        if n not in m.NodesSources:
            pMax = Nodes[n]['pMax'] / scale['p']
            pMin = Nodes[n]['pMin'] / scale['p']
            for t in m.Times:
                m.node_p[n, t].setlb(pMin)
                m.node_p[n, t].setub(pMax)

    # Mass balance
    def NODE_mass_balance_rule(m, n, t):
        IN = sum(m.outlet_w[arc, t] for arc in m.Nodes_ArcsIN[n])
        OUT = sum(m.inlet_w[arc, t] for arc in m.Nodes_ArcsOUT[n])
        cons = m.wCons[n, 0, t]
        return IN == OUT + cons
    
    m.NODE_mass_balance = pyo.Constraint(
        m.Nodes - m.NodesSources, m.Times,
        rule = NODE_mass_balance_rule
    )

    return m

''' Source nodes '''
def NODE_SOURCE_constr(m):
    
    # Pressure
    def NODE_SOURCE_pressure_rule(m, n, t):
        return m.node_p[n, t] == m.pSource[n, t]
    m.NODE_SOURCE_pressure = pyo.Constraint(
        m.NodesSources, m.Times,
        rule = NODE_SOURCE_pressure_rule
    )

    # Mass balance
    def NODE_SOURCE_mass_balance_rule(m, n, t): 
        IN = sum(m.outlet_w[arc, t] for arc in m.Nodes_ArcsIN[n])
        OUT = sum(m.inlet_w[arc, t] for arc in m.Nodes_ArcsOUT[n])    
        injection = m.wSource[n, t]
        return IN + injection == OUT
    
    m.NODE_SOURCE_mass_balance = pyo.Constraint(
        m.NodesSources, m.Times, 
        rule = NODE_SOURCE_mass_balance_rule
    )

    return m


#==============================================================================
""" CONSTRAINTS - STATION """
#==============================================================================

def STATION_constr(m, scale, inputData):
    def STATION_mass_balance_rule(m, s, t):
        return m.inlet_w[s, t] == m.outlet_w[s, t]
    
    m.STATION_mass_balance = pyo.Constraint(
        m.Stations, m.Times, 
        rule = STATION_mass_balance_rule) 

    def STATION_pressure_ratio_rule(m, s, t):
        n_out = m.Arcs_NodeOUT[s]
        n_in = m.Arcs_NodeIN[s]
        pout = m.node_p[n_out, t]
        pin = m.node_p[n_in, t]
        return pin*m.compressor_beta[s, t] == pout
    
    m.STATION_pressure_ratio = pyo.Constraint(
        m.Stations, m.Times, 
        rule = STATION_pressure_ratio_rule
    )

    def STATION_power_rule(m, s, t):
        w = m.inlet_w[s, t] * scale['w']
        P = m.compressor_P[s, t]
        beta = m.compressor_beta[s, t]
        eta = m.compressor_eta[s, t]
        cp = 2.34 # Cp gas used in IDAES
        Tin = inputData['GasParams']['Temperature'][0]
        gamma = inputData['GasParams']['gamma'][0]
        teta = 1 - 1 / gamma
        dh = cp * Tin * (beta ** teta - 1)
        return P == (w * dh / eta) / scale['P']
    m.STATION_power_balance = pyo.Constraint(
        m.Stations, m.Times, 
        rule = STATION_power_rule)
    
    return m


#==============================================================================
""" CONSTRAINTS - VALVE """
#==============================================================================

def VALVE_constr(m):

    def VALVE_mass_balance_rule(m, s, t):
        return m.inlet_w[s, t] == m.outlet_w[s, t]
    
    m.VALVE_mass_balance = pyo.Constraint(
        m.Valves, m.Times,
        rule = VALVE_mass_balance_rule
    )

    def VALVE_pressure_flow_rule(m, s, t):
        # if dp < 0 --> w = 0
        w = m.inlet_w[s, t]
        nout = m.Arcs_NodeOUT[s]
        pout = m.node_p[nout, t]
        nin = m.Arcs_NodeIN[s]
        pin = m.node_p[nin, t]
        dp = (pin - pout)
        return w * dp >= 0 # to ensure flow direction is the same as delta_p
    
    m.VALVE_pressure_flow = pyo.Constraint(
        m.Valves, m.Times,
        rule = VALVE_pressure_flow_rule)

    # m.inlet_w.setlb(0)

    return m


# ========================================================================
''' Constraints for pipes '''
# ========================================================================

''' Mass balance '''
# Use backward <=> use current state to evaluate mass flow rates
# Use forward <=> use previous state to evaluate mass flow rates
def PIPE_mass_constr(
        m, scale, dt, method = 'BACKWARD', dynamic=True):
    
    # assert: options
    assert(method in ['BACKWARD', 'FORWARD'])

    # =============== dynamic mass balance =========================
    def PIPE_mass_balance_dynamic_rule(
            m, p, vol, t): # vol= 1, 2,.., N-1
        # skip first time step: no t_prev   
        if t == m.Times.first():
            return pyo.Constraint.Skip
        else:
            V = m.Area[p] * m.Length[p]
            N = m.Nvol[p]    
            # LHS: derivative term (between t and t_prev)                         
            dMdt = Pipe_DerMass(
                m, p, vol, t, N, V, scale, dt)                
            # RHS: define timestep depending on differentiation method
            t_RHS = m.Times.prev(t) if method=="FORWARD" else t           
            # RHS: mass flow rates
            w = Pipe_MassFlow(m, p, vol, t_RHS, N)
            w_next = Pipe_MassFlow(m, p, vol+1, t_RHS, N) # next finite volume
            # RHS: consumption
            cons = m.wCons[p, vol, t_RHS]
    
            # constraint              
            return dMdt == (w*scale["w"] - w_next*scale["w"] - cons*scale["w"]) # w = in, w_next = out
        
    # =============== steady mass balance =========================
    def PIPE_mass_balance_steady_rule(
            m, p, vol, t): # vol= 1, 2,.., N-1 
        # RHS: mass flow rates
        N = m.Nvol[p]    
        w = Pipe_MassFlow(m, p, vol, t, N)
        w_next = Pipe_MassFlow(m, p, vol+1, t, N) # next finite volume
        # RHS: consumption
        cons = m.wCons[p, vol, t]

        # constraint              
        return 0 == (w*scale["w"] - w_next*scale["w"] - cons*scale["w"]) # w = in, w_next = out
    
    if dynamic:
        m.PIPE_mass_balance = pyo.Constraint(
            m.Pipes_VolCenterC, m.Times,
            rule = PIPE_mass_balance_dynamic_rule)
    else:
        m.PIPE_mass_balance = pyo.Constraint(
            m.Pipes_VolCenterC, m.Times,
            rule = PIPE_mass_balance_steady_rule)

    return m

''' Momentum balance '''
def PIPE_momentum_constr(
        m, scale, networkData):
    
    Nodes = networkData["Nodes"]

    # function: friction
    def friction_term(m, p, vol, t):
        # volumn length
        N = m.Nvol[p]
        start = 1
        end = m.Nvol[p] + 1
        l = m.Length[p] / (N-1)
        # first and last volume: half length
        if vol == start or (vol+1) == end:
            l = l / 2
        # rho friction
        rhoFrict = m.pipe_rho[p, vol, t] # Variable
        # u squared
        u2 = m.u2[p, vol, t]
        # Parameter
        A = m.Area[p]
        rough = m.Roughness[p]
        D = m.Diam[p]
        cf = (-4*pyo.log10(rough/(3.71 * D))) ** (-2) # Darcy-Weisbach friction factor, Colebrook-White equation
        omega = m.Omega[p]
        # final term
        coeffU2 = cf/2 * rhoFrict * (omega * l / A) * (scale["u2"]/scale["p"])
        return u2 * coeffU2 

    # function: geodetic: gravity term, from height difference between inlet and outlet of finite volume
    def geodetic_term(m, p, vol, t):      
        # find geodetic term  
        N = m.Nvol[p]
        start = 1
        end = N + 1
        n_in = m.Arcs_NodeIN[p]
        n_out = m.Arcs_NodeOUT[p]  
        dz = (Nodes[n_out]["height"]-Nodes[n_in]["height"])/(N-1)
        # first and last volume RCR --> R/2
        if vol == start or (vol+1) == end: # R/2
            dz = dz/2
        # geodetic head
        if int(dz) < 1:
            geodetic = 0     
        else: 
            rhoGeo = m.pipe_rho[p, vol, t]
            geodetic = rhoGeo * 9.81 * dz / scale["p"] 
        return geodetic  

    # constraint: momentum balance
    def PIPE_momentum_balance_rule(m, p, vol, t): # 1-->N     
        N = m.Nvol[p]       
        end = N + 1
        if vol == end: # skip if last volume
            return pyo.Constraint.Skip
        else:            
            # LHS
            press = Pipe_Pressure(m, p, vol, t, N)
            press_next = Pipe_Pressure(m, p, vol+1, t, N) # next finite volume
            # RHS
            friction = friction_term(m, p, vol, t)
            geodetic = geodetic_term(m, p, vol, t)                          
            return 100 * (press - press_next) == (friction + geodetic) * 100
    m.PIPE_momentum_balance = pyo.Constraint(
        m.Pipes_VolExtrR,
        m.Times, rule = PIPE_momentum_balance_rule)

    return m

''' Auxiliary constraints for nonlinear terms in momentum balance '''
def PIPE_nlp_auxiliary_constr( m, scale):
    # gas speed
    def PIPE_nonlinear_speed_rule(m, p, vol, t): 
        # flow
        w = Pipe_MassFlow(m, p, vol, t, m.Nvol[p])
        # density
        rho = m.pipe_rho[p, vol, t]
        A = m.Area[p]
        return m.u[p, vol, t] * rho == (w * scale["w"] / A)
    m.PIPE_nonlinear_speed = pyo.Constraint(
        m.Pipes_VolExtrC, m.Times, 
        rule = PIPE_nonlinear_speed_rule)

    # gas density
    def PIPE_nonlinear_rho_rule(m, p, vol, t): 
        # gas density (equation of state)
        N = m.Nvol[p]
        rhoCalc = Pipe_GasDensity(m, scale, p, vol, t, N, calc = True) 
        return m.pipe_rho[p, vol, t] == rhoCalc
    m.PIPE_nonlinear_rho = pyo.Constraint(
        m.Pipes_VolExtrR, m.Times, 
        rule = PIPE_nonlinear_rho_rule)
    
    return m

'''Give positive/negative flow direction a smooth transition to avoid numerical issues in flow reversal'''
def PIPE_flow_reversal_constr(m, scale):
    # smoothing flow reversal term
    eps = m.eps
    def PIPE_nonlinear_friction_smooth_rule(m, p, vol, t): 
        u = m.u[p, vol, t]
        abs_u = pyo.sqrt(u**2 + eps)
        return m.u2[p, vol, t] == (u * abs_u) / scale["u2"]
    m.PIPE_nonlinear_friction_smooth = pyo.Constraint(
        m.Pipes_VolExtrC, m.Times, 
        rule = PIPE_nonlinear_friction_smooth_rule)
    
    return m

#=========================================================================
""" FIX INPUTS """
#==============================================================================
''' Scale input data from SI to dimensionless, assign to Pyomo variables, and fix wCons '''
def fix_exogenous_inputs(m, scale, Options, networkData, inputData):

    for k in ['pSource', 'wSource', 'wCons']:
        # define wheter to fix or not variable
        fix = True # w_consume should be fixed
        if k == 'wSource' or k == 'pSource':
            fix = False

        # define scale factor (from SI base units)
        scale_factor = 1
        if k in ['wSource', 'wCons']:
            scale_factor = scale['w']
        elif k in ['pSource']:
            scale_factor = scale['p']
        # retrieve corresponding pyomo object and indices from the variable name: getattr(m, 'wCons')  ≡  m.wCons
        varobject = getattr(m, k)
        indices = list(varobject)

        # set values
        # iterate over all indices of the Pyomo variable
        # e.g. indices = [('source_1', 0), ('source_1', 1), ..., ('source_2', 0), ...]
        for indx in indices:
            t = indx[-1] # last element is time, e.g. indx = ('source_1', 0) → t = 0
            compo0 = indx[0] # first element is node/pipe number, e.g. indx = ('source_1', 0) → compo0 = 'source_1'
            # if the variable is pSource or wSource, map the Pyomo node name
            # to the corresponding source name used in inputData
            # e.g. networkData['Nodes']['source_1']['source'] = 'source_1'
            # (names may differ between networkData and inputData)

            # e.g. compo0 = 'source_1' → 'source_1' (unchanged in this dataset,
            #      but handles cases where Pyomo name ≠ inputData name)
            if 'Source' in k: # pSource or wSource
                source_name = networkData['Nodes'][compo0]['source']
                compo0 = compo0.replace(compo0, source_name) # compo0 = source_name
        
            # two-level index: (node, time) → pSource, wSource
            # e.g. inputData['pSource']['source_1'][0] = 4101325 Pa
            if len(indx) == 2:
                value = inputData[k][compo0][t]

            # three-level index: (pipe, vol, time) → wCons only
            # e.g. inputData['wCons']['pipe_1'][2][0] = 0.5 kg/s
            else:
                compo1 = indx[1] # control volume index, e.g. 2
                value = inputData[k][compo0][compo1][t]

            # scale from SI units to dimensionless for IPOPT
            # pSource: 4101325 Pa / 1e6 = 4.101325
            # wSource: 158.0903 kg/s / 100 = 1.580903
            # wCons:   0.5 kg/s / 100 = 0.00
            value = value / scale_factor

            # write dimensionless value into the Pyomo variable
            # e.g. m.pSource['source_1', 0] = 4.101325
            varobject[indx] = value

        # fix the variable if required (only wCons is fixed)
        # fix=True  → varobject.fix(): IPOPT cannot change this value
        # fix=False → pSource/wSource remain free decision variables
        #             IPOPT can adjust within bounds [pMin, pMax] = [101325, 8101325] Pa
        if fix:
            varobject.fix()

    return m

''' Initialization of network to default values '''
def init_network_default(m, p_default = 55e5):

    scale = {
        "p": 100000, # pressure
        "P": 100000, # compressor power
        "w": 1, # mass flow
        'u2': 10 # squared speed
        }

    # pressure initialization
    m.node_p[:, :] = p_default / scale['p']
    m.interm_p[:, :, :] = p_default / scale['p']

    # ! add other default initialization
    
    return m

#==============================================================================
""" BUILD MODEL """
#==============================================================================

def buildNonLinearModel(
        networkData, inputData, Opt, duals=False):
    
    # scale dictionary --> scales variables (variables are all defined in SI base units)
    scale = {
        "p": 100000, # pressure
        "P": 100000, # compressor power
        "w": 1, # mass flow
        'u2': 10 # squared speed
        }
    
    # Model
    m = pyo.ConcreteModel('gasnetwork')

    ''' Sets '''
    # Time
    m = TIME_sets(m, Opt)
    # Network
    m = NODE_sets(m, networkData)
    m = ARC_sets(m, networkData)
    # Station
    m = STATION_set(m, networkData)
    # Pipes
    m = PIPE_sets(m, networkData)
    # Valves
    m = VALVE_sets(m, networkData)

    ''' Parameters '''
    # Gas parameters
    m = PIPE_gas_params(m, networkData)
    # Geometry
    m = PIPE_geom_params(m, networkData)
    # Smoothing
    m = PIPE_smoothing_params(m, Opt['eps'])

    ''' Variables '''
    # Network
    m = NODE_vars(m)
    m = ARC_vars(m)
    # Station
    m = STATIONS_vars(m)
    # Pipes
    m = PIPE_vars(m, scale, networkData)
    # Duals
    if duals:
        m = DUALS_ipopt_vars(m)
    # FIX EXOGENOUS INPUTS (TIME varying input variables)
    # ! FIX EXOGENOUS INPUTS ONCE DATA IS AVAILABLE
    m = fix_exogenous_inputs(m, scale, Opt, networkData, inputData)

    ''' Objective '''
    m = OBJ_compressor_power(m, Opt)

    ''' Constraints '''
    # Nodes
    m = NODE_constr(m, scale, networkData)
    m = NODE_SOURCE_constr(m) # supply nodes
    # Station
    m = STATION_constr(m, scale, inputData)
    # Valve
    m = VALVE_constr(m)
    # PIPE
    m = PIPE_mass_constr(
        m, scale, Opt['dt'],
        method=Opt['finite_diff_time'], dynamic=Opt['dynamic']
    )
    m = PIPE_momentum_constr(m, scale, networkData)
    m = PIPE_nlp_auxiliary_constr(m, scale)
    m = PIPE_flow_reversal_constr(m, scale)

    return m


''' test '''
if __name__ == "__main__":
    import json
    import os

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test')

    with open(os.path.join(path, 'Options.json'), 'r') as f:
        Opt = json.load(f)

    from gas_net.util.import_data import import_data_from_excel

    networkData, inputData = import_data_from_excel(
        os.path.join(path, 'networkData.xlsx'),
        os.path.join(path, 'inputData.xlsx')
    )

    try:
        m = buildNonLinearModel(networkData, inputData, Opt)
        print("Model success")
    except Exception as e:
        print(f"Error: {e}")
