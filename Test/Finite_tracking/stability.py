import pyomo.environ as pyo


#==============================================================================
""" stability_constraint """
#==============================================================================

def apply_stability_constraint(m_controller):
    m_controller.lyapunov_function_current = pyo.Var(initialize=1)
    m_controller.lyapunov_function_prev = pyo.Param(initialize=1, mutable=True)
    m_controller.tracking_cost_plant_prev = pyo.Param(initialize=1, mutable=True)
    m_controller.delta = pyo.Param(initialize=0.1)

    def _lyapunov_function_definition(m):
        return m.lyapunov_function_current == (
            sum(
                (m.interm_p[p, vol, t] - m.interm_p_ocss[p, vol, t]) ** 2
                for p, vol in m.Pipes_VolExtrR_interm
                for t in m.Times
                if t != m.Times.last()
            )
            + sum(
                (m.compressor_P[s, t] - m.compressor_P_ocss[s, t]) ** 2
                for s in m.Stations
                for t in m.Times
                if t != m.Times.last()
            )
            # + sum(
            #     (m.compressor_beta[s, t] - m.compressor_beta_ocss[s, t]) ** 2
            #     for s in m.Stations
            #     for t in m.Times
            #     if t != m.Times.last()
            # )
        )

    m_controller.lyapunov_function_definition = pyo.Constraint(
        rule=_lyapunov_function_definition
    )

    def _stability_constraint(m):
        return (
            m.lyapunov_function_current
            <= m.lyapunov_function_prev
            - m.delta * m.tracking_cost_plant_prev
        )

    m_controller.stability_constraint = pyo.Constraint(
        rule=_stability_constraint
    )
    return m_controller


def update_tracking_cost_plant_prev(
        m_controller, m_plant, plant_tf, reference_t):
    tracking_cost = sum(
        (
            m_plant.interm_p[p, vol, plant_tf]
            - m_controller.interm_p_ocss[p, vol, reference_t]
        ) ** 2
        for p, vol in m_controller.Pipes_VolExtrR_interm
    ) + sum(
        (
            m_plant.compressor_P[s, plant_tf]
            - m_controller.compressor_P_ocss[s, reference_t]
        ) ** 2
        for s in m_controller.Stations
    )
    # + sum(
    #     (
    #         m_plant.compressor_beta[s, plant_tf]
    #         - m_controller.compressor_beta_ocss[s, reference_t]
    #     ) ** 2
    #     for s in m_controller.Stations
    # )
    m_controller.tracking_cost_plant_prev.set_value(
        pyo.value(tracking_cost)
    )
