"""
Reproduce the main hinged-wing gust-response results from the paper
description, without using the authors' saved .mat data.

The implementation follows the linearized equations and parameter values
reported in Stevenson et al. (2023), Royal Society Open Science 10:221607.
It writes CSV time histories and lightweight SVG plots to
outputs_from_description/.
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


OUT = Path("outputs_from_description")


@dataclass(frozen=True)
class Params:
    g: float = 9.81
    chord: float = 0.15
    span: float = 0.4
    total_mass: float = 0.3
    rho: float = 1.2
    speed: float = 8.0
    lift_slope: float = 2.0 * math.pi
    gust_length: float = 1.4
    gust_peak: float = 2.4
    gust_on: float = 0.1
    sample_dt: float = 0.005
    solve_dt: float = 0.0005
    stations: int = 50

    @property
    def area(self) -> float:
        return self.chord * self.span

    @property
    def dynamic_pressure(self) -> float:
        return 0.5 * self.rho * self.speed**2

    @property
    def gust_duration(self) -> float:
        return self.gust_length / self.speed

    @property
    def end_time(self) -> float:
        return self.gust_on + self.gust_duration

    @property
    def alpha0(self) -> float:
        return self.total_mass * self.g / (
            self.dynamic_pressure * 2.0 * self.area * self.lift_slope
        )


def wing_mass_properties(params: Params, wing_mass_fraction: float) -> dict[str, float]:
    single_wing_mass = wing_mass_fraction * params.total_mass / 2.0
    body_mass = params.total_mass * (1.0 - wing_mass_fraction)

    # Paper equation (2.10): triangular/linear mass density,
    # m'(y) = -(2 mw / b) (y / b - 1).
    lw = params.span / 3.0
    ih = single_wing_mass * params.span**2 / 6.0
    percussion = ih / (single_wing_mass * lw)

    return {
        "mw": single_wing_mass,
        "mb": body_mass,
        "lw": lw,
        "ih": ih,
        "percussion": percussion,
    }


def gust(t: float, params: Params) -> float:
    if t < params.gust_on or t > params.end_time:
        return 0.0
    phase = (t - params.gust_on) / params.gust_duration
    return 0.5 * params.gust_peak * (1.0 - math.cos(2.0 * math.pi * phase))


def lift_curve_llc(aoa: np.ndarray | float, params: Params) -> np.ndarray | float:
    return params.lift_slope * aoa


def lift_curve_nlc(aoa: np.ndarray | float, params: Params) -> np.ndarray | float:
    # The paper's soft-stall approximation: cL rises linearly until cL = 1,
    # then remains on a flat plateau.
    return np.minimum(params.lift_slope * aoa, 1.0)


def gust_loads(
    z_dot: float,
    theta: float,
    theta_dot: float,
    t: float,
    params: Params,
    lift_curve: Callable[[np.ndarray, Params], np.ndarray],
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    y = np.linspace(0.0, params.span, params.stations)
    dalpha = (gust(t, params) - z_dot - y * theta_dot) / params.speed
    cl = lift_curve(params.alpha0 + dalpha, params)
    force_per_span = params.dynamic_pressure * params.chord * cl
    force = float(np.trapezoid(force_per_span, y))
    moment = float(np.trapezoid(y * force_per_span, y))
    centre = moment / force if abs(force) > 1e-12 else float("nan")
    return force, moment, centre, dalpha, force_per_span


def locked_rhs(
    t: float,
    state: np.ndarray,
    params: Params,
    lift_curve: Callable[[np.ndarray, Params], np.ndarray],
) -> np.ndarray:
    _, z_dot = state
    force, _, _, _, _ = gust_loads(z_dot, 0.0, 0.0, t, params, lift_curve)
    z_ddot = (2.0 * force - params.total_mass * params.g) / params.total_mass
    return np.array([z_dot, z_ddot])


def unlocked_rhs(
    t: float,
    state: np.ndarray,
    params: Params,
    wing_mass_fraction: float,
    hinge_stiffness: float,
    lift_curve: Callable[[np.ndarray, Params], np.ndarray],
) -> np.ndarray:
    z, z_dot, theta, theta_dot = state
    props = wing_mass_properties(params, wing_mass_fraction)
    force, moment, _, _, _ = gust_loads(z_dot, theta, theta_dot, t, params, lift_curve)

    l_force0 = 0.5 * params.span
    static_torque = (
        params.total_mass * params.g * l_force0 / 2.0
        - props["mw"] * params.g * props["lw"]
    )
    torque = static_torque + hinge_stiffness * theta

    # Linearized version of the authors' two-DOF equations, solved for
    # z_ddot and theta_ddot.
    denominator = params.total_mass * props["ih"] - 2.0 * (props["mw"] * props["lw"]) ** 2
    rhs_force = 2.0 * force - params.total_mass * params.g
    rhs_moment = moment - torque - props["mw"] * params.g * props["lw"]
    z_ddot = (props["ih"] * rhs_force - 2.0 * props["mw"] * props["lw"] * rhs_moment) / denominator
    theta_ddot = (
        params.total_mass * rhs_moment - props["mw"] * props["lw"] * rhs_force
    ) / denominator

    return np.array([z_dot, z_ddot, theta_dot, theta_ddot])


def rk4(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    initial_state: Iterable[float],
    params: Params,
) -> tuple[np.ndarray, np.ndarray]:
    dt = params.solve_dt
    steps = int(round(params.end_time / dt))
    t = np.linspace(0.0, steps * dt, steps + 1)
    states = np.zeros((steps + 1, len(list(initial_state))))
    states[0] = np.array(list(initial_state), dtype=float)
    for i in range(steps):
        ti = t[i]
        x = states[i]
        k1 = rhs(ti, x)
        k2 = rhs(ti + 0.5 * dt, x + 0.5 * dt * k1)
        k3 = rhs(ti + 0.5 * dt, x + 0.5 * dt * k2)
        k4 = rhs(ti + dt, x + dt * k3)
        states[i + 1] = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    sample_every = max(1, int(round(params.sample_dt / dt)))
    return t[::sample_every], states[::sample_every]


def simulate_locked(
    params: Params, lift_curve: Callable[[np.ndarray, Params], np.ndarray]
) -> dict[str, np.ndarray]:
    t, state = rk4(lambda ti, x: locked_rhs(ti, x, params, lift_curve), [0.0, 0.0], params)
    return {"t": t, "z": state[:, 0], "z_dot": state[:, 1]}


def simulate_unlocked(
    params: Params,
    wing_mass_fraction: float,
    hinge_stiffness: float,
    lift_curve: Callable[[np.ndarray, Params], np.ndarray],
) -> dict[str, np.ndarray | float]:
    t, state = rk4(
        lambda ti, x: unlocked_rhs(
            ti, x, params, wing_mass_fraction, hinge_stiffness, lift_curve
        ),
        [0.0, 0.0, 0.0, 0.0],
        params,
    )
    return {
        "t": t,
        "z": state[:, 0],
        "z_dot": state[:, 1],
        "theta": state[:, 2],
        "theta_dot": state[:, 3],
        "muw": wing_mass_fraction,
        "kt": hinge_stiffness,
    }


def enrich_solution(
    solution: dict[str, np.ndarray | float],
    params: Params,
    lift_curve: Callable[[np.ndarray, Params], np.ndarray],
    mode: str,
) -> dict[str, np.ndarray | float]:
    t = solution["t"]
    force = []
    moment = []
    centre = []
    gust_values = []
    for i, ti in enumerate(t):
        z_dot = float(solution["z_dot"][i])
        theta = float(solution["theta"][i]) if mode == "unlocked" else 0.0
        theta_dot = float(solution["theta_dot"][i]) if mode == "unlocked" else 0.0
        f, mf, cp, _, _ = gust_loads(z_dot, theta, theta_dot, float(ti), params, lift_curve)
        force.append(f)
        moment.append(mf)
        centre.append(cp)
        gust_values.append(gust(float(ti), params))
    out = dict(solution)
    out["force"] = np.array(force)
    out["moment"] = np.array(moment)
    out["centre"] = np.array(centre)
    out["gust"] = np.array(gust_values)
    return out


def body_acceleration_from_velocity(t: np.ndarray, z_dot: np.ndarray) -> np.ndarray:
    return np.gradient(z_dot, t)


def centre_of_mass_velocity(
    solution: dict[str, np.ndarray | float], params: Params, wing_mass_fraction: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z_dot = solution["z_dot"]
    if "theta_dot" not in solution:
        return z_dot, z_dot, z_dot
    props = wing_mass_properties(params, wing_mass_fraction)
    theta = solution["theta"]
    theta_dot = solution["theta_dot"]
    body = z_dot
    wing = z_dot + props["lw"] * theta_dot * np.cos(theta)
    system = z_dot + wing_mass_fraction * props["lw"] * theta_dot * np.cos(theta)
    return body, wing, system


def write_csv(path: Path, columns: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        names = list(columns)
        writer.writerow(names)
        for row in zip(*(columns[name] for name in names)):
            writer.writerow([f"{float(value):.10g}" for value in row])


def make_svg_grid(
    path: Path,
    panels: list[dict],
    width: int = 1100,
    height: int = 850,
) -> None:
    margin_left = 90
    margin_bottom = 70
    panel_w = 420
    panel_h = 280
    gap_x = 120
    gap_y = 130
    origins = [
        (margin_left, 60),
        (margin_left + panel_w + gap_x, 60),
        (margin_left, 60 + panel_h + gap_y),
        (margin_left + panel_w + gap_x, 60 + panel_h + gap_y),
    ]

    def points(xs: np.ndarray, ys: np.ndarray, origin: tuple[int, int], xlim, ylim) -> str:
        x0, y0 = origin
        xx = x0 + (xs - xlim[0]) / (xlim[1] - xlim[0]) * panel_w
        yy = y0 + panel_h - (ys - ylim[0]) / (ylim[1] - ylim[0]) * panel_h
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xx, yy))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,Helvetica,sans-serif;font-size:22px}.small{font-size:17px}.axis{stroke:#222;stroke-width:2;fill:none}.grid{stroke:#ddd;stroke-width:1}.legend{font-size:20px}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for idx, panel in enumerate(panels):
        origin = origins[idx]
        x0, y0 = origin
        xlim, ylim = panel["xlim"], panel["ylim"]
        parts.append(f'<rect x="{x0}" y="{y0}" width="{panel_w}" height="{panel_h}" fill="none" class="axis"/>')
        for frac in (0.0, 0.5, 1.0):
            xt = x0 + frac * panel_w
            yt = y0 + panel_h - frac * panel_h
            xv = xlim[0] + frac * (xlim[1] - xlim[0])
            yv = ylim[0] + frac * (ylim[1] - ylim[0])
            parts.append(f'<line x1="{xt:.1f}" x2="{xt:.1f}" y1="{y0}" y2="{y0 + panel_h}" class="grid"/>')
            parts.append(f'<line x1="{x0}" x2="{x0 + panel_w}" y1="{yt:.1f}" y2="{yt:.1f}" class="grid"/>')
            parts.append(f'<text x="{xt - 25:.1f}" y="{y0 + panel_h + 30}" class="small">{xv:.2g}</text>')
            parts.append(f'<text x="{x0 - 55}" y="{yt + 7:.1f}" class="small">{yv:.2g}</text>')
        parts.append(f'<text x="{x0 + panel_w / 2 - 55}" y="{y0 + panel_h + margin_bottom - 10}">{panel["xlabel"]}</text>')
        parts.append(f'<text x="{x0 - 70}" y="{y0 - 12}" transform="rotate(-90 {x0 - 70},{y0 - 12})">{panel["ylabel"]}</text>')
        parts.append(f'<text x="{x0 + 8}" y="{y0 + 28}" class="small">({chr(97 + idx)})</text>')

        if "shade" in panel:
            xs, ys = panel["shade"]
            polygon = (
                f"{x0},{y0 + panel_h} "
                + points(xs, ys, origin, xlim, ylim)
                + f" {x0 + panel_w},{y0 + panel_h}"
            )
            parts.append(f'<polygon points="{polygon}" fill="#d7d7d7" opacity="0.75"/>')

        for line in panel["lines"]:
            dash = ' stroke-dasharray="8 6"' if line.get("dash") else ""
            width_line = line.get("width", 4)
            parts.append(
                f'<polyline points="{points(line["x"], line["y"], origin, xlim, ylim)}" '
                f'fill="none" stroke="{line["color"]}" stroke-width="{width_line}"{dash}/>'
            )
        lx = x0 + 25
        ly = y0 + 35
        for line in panel["lines"][:5]:
            if not line.get("label"):
                continue
            dash = ' stroke-dasharray="8 6"' if line.get("dash") else ""
            parts.append(f'<line x1="{lx}" x2="{lx + 55}" y1="{ly}" y2="{ly}" stroke="{line["color"]}" stroke-width="5"{dash}/>')
            parts.append(f'<text x="{lx + 65}" y="{ly + 7}" class="legend">{line["label"]}</text>')
            ly += 28
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")


def figure3(params: Params) -> None:
    f0 = params.total_mass * params.g / 2.0
    fixed = enrich_solution(simulate_locked(params, lift_curve_llc), params, lift_curve_llc, "locked")
    hinged = enrich_solution(simulate_unlocked(params, 0.2, 0.0, lift_curve_llc), params, lift_curve_llc, "unlocked")
    t = fixed["t"]
    ts = t - params.gust_on
    body, wing, system = centre_of_mass_velocity(hinged, params, 0.2)
    fixed_body, _, fixed_system = centre_of_mass_velocity(fixed, params, 0.2)
    dr_hinged = wing_mass_properties(params, 0.2)["mb"] * body_acceleration_from_velocity(t, hinged["z_dot"]) / 2.0
    df_fixed = fixed["force"] - f0
    df_hinged = hinged["force"] - f0
    immobile_force = np.array([gust_loads(0, 0, 0, ti, params, lift_curve_llc)[0] for ti in t]) - f0
    aerodynamic = fixed_system - system
    inertial = system - body
    potential = fixed_system

    write_csv(
        OUT / "figure3_from_description.csv",
        {
            "time_shifted_s": ts,
            "fixed_force_increment_N": df_fixed,
            "hinged_force_increment_N": df_hinged,
            "hinged_reaction_increment_N": dr_hinged,
            "fixed_fuselage_velocity_m_s": fixed_body,
            "hinged_fuselage_velocity_m_s": body,
            "hinged_centre_pressure_over_span": hinged["centre"] / params.span,
        },
    )
    make_svg_grid(
        OUT / "figure3_from_description.svg",
        [
            {
                "xlim": (-0.05, 0.13),
                "ylim": (-1, 5),
                "xlabel": "Time (s)",
                "ylabel": "Force / reaction (N)",
                "shade": (ts, immobile_force),
                "lines": [
                    {"x": ts, "y": immobile_force, "color": "#000", "label": "Immobile", "width": 5},
                    {"x": ts, "y": df_fixed, "color": "#3498db", "label": "Fixed", "width": 3},
                    {"x": ts, "y": df_hinged, "color": "#9fc32b", "label": "Hinged force", "width": 3},
                    {"x": ts, "y": dr_hinged, "color": "#9fc32b", "label": "Hinged reaction", "width": 6},
                ],
            },
            {
                "xlim": (-0.05, 0.13),
                "ylim": (-0.2, 1.5),
                "xlabel": "Time (s)",
                "ylabel": "Mass velocity (m/s)",
                "lines": [
                    {"x": ts, "y": fixed_system, "color": "#3498db", "label": "Fixed"},
                    {"x": ts, "y": system, "color": "#9fc32b", "label": "Hinged"},
                    {"x": ts, "y": wing, "color": "#9fc32b", "label": "Wings", "dash": True},
                    {"x": ts, "y": body, "color": "#9fc32b", "label": "Body", "dash": True},
                ],
            },
            {
                "xlim": (-0.05, 0.13),
                "ylim": (0, 1),
                "xlabel": "Time (s)",
                "ylabel": "Centre of pressure",
                "lines": [
                    {"x": ts, "y": fixed["centre"] / params.span, "color": "#3498db", "label": "Fixed"},
                    {"x": ts, "y": hinged["centre"] / params.span, "color": "#9fc32b", "label": "Hinged"},
                ],
            },
            {
                "xlim": (-0.05, 0.13),
                "ylim": (-0.2, 1.6),
                "xlabel": "Time (s)",
                "ylabel": "Rejection (m/s)",
                "lines": [
                    {"x": ts, "y": potential, "color": "#000", "label": "Potential"},
                    {"x": ts, "y": aerodynamic, "color": "#9fc32b", "label": "Aerodynamic"},
                    {"x": ts, "y": inertial, "color": "#4b3f99", "label": "Inertial"},
                ],
            },
        ],
    )


def figure4(params: Params) -> None:
    fixed_nlc = enrich_solution(simulate_locked(params, lift_curve_nlc), params, lift_curve_nlc, "locked")
    hinged_llc = enrich_solution(simulate_unlocked(params, 0.2, 0.0, lift_curve_llc), params, lift_curve_llc, "unlocked")
    hinged_nlc = enrich_solution(simulate_unlocked(params, 0.2, 0.0, lift_curve_nlc), params, lift_curve_nlc, "unlocked")
    t = fixed_nlc["t"]
    ts = t - params.gust_on
    mb = wing_mass_properties(params, 0.2)["mb"]
    react_llc = mb * body_acceleration_from_velocity(t, hinged_llc["z_dot"]) / 2.0
    react_nlc = mb * body_acceleration_from_velocity(t, hinged_nlc["z_dot"]) / 2.0
    body_nlc, _, system_nlc = centre_of_mass_velocity(hinged_nlc, params, 0.2)
    fixed_body, _, fixed_system = centre_of_mass_velocity(fixed_nlc, params, 0.2)
    aerodynamic = fixed_system - system_nlc
    inertial = system_nlc - body_nlc

    write_csv(
        OUT / "figure4_from_description.csv",
        {
            "time_shifted_s": ts,
            "fixed_nlc_velocity_m_s": fixed_body,
            "hinged_llc_velocity_m_s": hinged_llc["z_dot"],
            "hinged_nlc_velocity_m_s": body_nlc,
            "hinged_nlc_centre_pressure_over_span": hinged_nlc["centre"] / params.span,
            "hinged_nlc_reaction_N": react_nlc,
        },
    )
    make_svg_grid(
        OUT / "figure4_from_description.svg",
        [
            {
                "xlim": (0, 25),
                "ylim": (0, 1.6),
                "xlabel": "AoA (deg)",
                "ylabel": "Lift coefficient",
                "lines": [
                    {
                        "x": np.linspace(0, 25, 200),
                        "y": lift_curve_nlc(np.deg2rad(np.linspace(0, 25, 200)), params),
                        "color": "#000",
                        "label": "NLC",
                    }
                ],
            },
            {
                "xlim": (-0.05, 0.13),
                "ylim": (-0.5, 1.8),
                "xlabel": "Time (s)",
                "ylabel": "Fuselage reaction (N)",
                "lines": [
                    {"x": ts, "y": (1 - 0.2) * (fixed_nlc["force"] - params.total_mass * params.g / 2), "color": "#3498db", "label": "Fixed NLC"},
                    {"x": ts, "y": react_llc, "color": "#999", "label": "Hinged LLC"},
                    {"x": ts, "y": react_nlc, "color": "#9fc32b", "label": "Hinged NLC"},
                ],
            },
            {
                "xlim": (-0.05, 0.13),
                "ylim": (0.2, 0.7),
                "xlabel": "Time (s)",
                "ylabel": "Centre of pressure",
                "lines": [
                    {"x": ts, "y": fixed_nlc["centre"] / params.span, "color": "#3498db", "label": "Fixed NLC"},
                    {"x": ts, "y": hinged_llc["centre"] / params.span, "color": "#999", "label": "Hinged LLC"},
                    {"x": ts, "y": hinged_nlc["centre"] / params.span, "color": "#9fc32b", "label": "Hinged NLC"},
                ],
            },
            {
                "xlim": (-0.05, 0.13),
                "ylim": (-0.2, 1.3),
                "xlabel": "Time (s)",
                "ylabel": "Rejection (m/s)",
                "lines": [
                    {"x": ts, "y": fixed_system, "color": "#000", "label": "Potential"},
                    {"x": ts, "y": aerodynamic, "color": "#9fc32b", "label": "Aerodynamic"},
                    {"x": ts, "y": inertial, "color": "#4b3f99", "label": "Inertial"},
                ],
            },
        ],
    )


def figure5(params: Params) -> None:
    fixed = enrich_solution(simulate_locked(params, lift_curve_nlc), params, lift_curve_nlc, "locked")
    t = fixed["t"]
    ts = t - params.gust_on
    masses = [0.2, 0.35, 0.5]
    sims = [enrich_solution(simulate_unlocked(params, muw, 0.0, lift_curve_nlc), params, lift_curve_nlc, "unlocked") for muw in masses]
    colours = ["#b6d33a", "#80b82d", "#3a8f2d"]

    csv_cols = {"time_shifted_s": ts, "fixed_velocity_m_s": fixed["z_dot"]}
    for muw, sim in zip(masses, sims):
        csv_cols[f"muw_{muw:g}_velocity_m_s"] = sim["z_dot"]
        csv_cols[f"muw_{muw:g}_centre_pressure_over_span"] = sim["centre"] / params.span
    write_csv(OUT / "figure5_from_description.csv", csv_cols)

    lines_cp = [{"x": ts, "y": fixed["centre"] / params.span, "color": "#3498db", "label": "Fixed"}]
    lines_vel = [{"x": ts, "y": fixed["z_dot"], "color": "#3498db", "label": "Fixed"}]
    lines_rej = []
    for muw, sim, colour in zip(masses, sims, colours):
        body, _, system = centre_of_mass_velocity(sim, params, muw)
        _, _, fixed_system = centre_of_mass_velocity(fixed, params, muw)
        lines_cp.append({"x": ts, "y": sim["centre"] / params.span, "color": colour, "label": f"{muw:.2f} M"})
        lines_vel.append({"x": ts, "y": body, "color": colour, "label": f"{muw:.2f} M"})
        lines_rej.append({"x": ts, "y": system - body, "color": colour, "label": f"Inertial {muw:.2f}"})
        lines_rej.append({"x": ts, "y": fixed_system - system, "color": colour, "dash": True, "label": f"Aero {muw:.2f}"})
    make_svg_grid(
        OUT / "figure5_from_description.svg",
        [
            {"xlim": (-0.05, 0.13), "ylim": (0.2, 0.65), "xlabel": "Time (s)", "ylabel": "Centre of pressure", "lines": lines_cp},
            {"xlim": (-0.05, 0.13), "ylim": (-0.05, 0.7), "xlabel": "Time (s)", "ylabel": "Fuselage velocity (m/s)", "lines": lines_vel},
            {"xlim": (-0.05, 0.13), "ylim": (-0.05, 0.7), "xlabel": "Time (s)", "ylabel": "Rejection (m/s)", "lines": lines_rej[:5]},
            {"xlim": (-0.05, 0.13), "ylim": (-0.05, 0.7), "xlabel": "Time (s)", "ylabel": "Rejection (m/s)", "lines": lines_rej[5:]},
        ],
    )


def figure6(params: Params) -> None:
    fixed = enrich_solution(simulate_locked(params, lift_curve_nlc), params, lift_curve_nlc, "locked")
    t = fixed["t"]
    ts = t - params.gust_on
    stiffnesses = [0.0, 0.1, 1.0, 10.0]
    colours = ["#226f31", "#78b82a", "#c4a51a", "#9b3333"]
    sims = [enrich_solution(simulate_unlocked(params, 0.5, kt, lift_curve_nlc), params, lift_curve_nlc, "unlocked") for kt in stiffnesses]
    csv_cols = {"time_shifted_s": ts, "fixed_velocity_m_s": fixed["z_dot"]}
    for kt, sim in zip(stiffnesses, sims):
        csv_cols[f"kt_{kt:g}_velocity_m_s"] = sim["z_dot"]
        csv_cols[f"kt_{kt:g}_theta_deg"] = np.rad2deg(sim["theta"])
    write_csv(OUT / "figure6_from_description.csv", csv_cols)
    lines_v = [{"x": ts, "y": fixed["z_dot"], "color": "#3498db", "label": "Fixed"}]
    lines_theta = []
    for kt, sim, colour in zip(stiffnesses, sims, colours):
        lines_v.append({"x": ts, "y": sim["z_dot"], "color": colour, "label": f"kt={kt:g}"})
        lines_theta.append({"x": ts, "y": np.rad2deg(sim["theta"]), "color": colour, "label": f"kt={kt:g}"})
    make_svg_grid(
        OUT / "figure6_from_description.svg",
        [
            {"xlim": (-0.05, 0.13), "ylim": (-0.05, 0.85), "xlabel": "Time (s)", "ylabel": "Fuselage velocity (m/s)", "lines": lines_v},
            {"xlim": (-0.05, 0.13), "ylim": (0, 35), "xlabel": "Time (s)", "ylabel": "Wing angle (deg)", "lines": lines_theta},
            {"xlim": (-0.05, 0.13), "ylim": (-0.05, 0.85), "xlabel": "Time (s)", "ylabel": "Fuselage velocity (m/s)", "lines": lines_v[1:]},
            {"xlim": (-0.05, 0.13), "ylim": (0, 35), "xlabel": "Time (s)", "ylabel": "Wing angle (deg)", "lines": lines_theta},
        ],
    )


def main() -> None:
    OUT.mkdir(exist_ok=True)
    params = Params()
    figure3(params)
    figure4(params)
    figure5(params)
    figure6(params)
    print(f"Wrote reproduction outputs to {OUT.resolve()}")


if __name__ == "__main__":
    main()
