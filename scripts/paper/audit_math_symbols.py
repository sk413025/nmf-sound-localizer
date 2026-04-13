#!/usr/bin/env python3
"""Generate a repeatable math-symbol consistency audit for manuscript surfaces.

This script scans the main manuscript and supplementary information via Pandoc
AST, extracts every math node, maps paper-local semantic symbols onto a
canonical registry, and writes a Markdown audit note.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
TARGETS = [
    REPO_ROOT / "paper" / "manuscript" / "manuscript.md",
    REPO_ROOT / "paper" / "manuscript" / "supplementary.md",
]
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "working-notes" / "math_symbol_consistency_audit.md"

FAMILY_ORDER = [
    "physical model",
    "measured fingerprints and calibration",
    "reduced surrogate",
    "grouped routing / solver",
    "statistics",
    "cross-object descriptors",
    "unclassified",
]
VERDICT_ORDER = {
    "collision": 0,
    "undefined": 1,
    "alias drift": 2,
    "main-only": 3,
    "supplement-only": 4,
    "consistent": 5,
}
NON_COLLISION_LABELS = {"Hx", "A x", "|H|", r"\|x\|_0"}


@dataclass(frozen=True)
class PatternSpec:
    label: str
    regex: str


@dataclass(frozen=True)
class SymbolSpec:
    key: str
    canonical: str
    family: str
    meaning: str
    recommended: str
    patterns: tuple[PatternSpec, ...]
    defining_patterns: tuple[str, ...] = ()
    note: str = ""
    drift_on_multiple_forms: bool = False


@dataclass
class MathOccurrence:
    source: str
    heading_path: str
    kind: str
    raw_tex: str
    label: str | None
    order: int

    def location(self) -> str:
        if self.label:
            return f"{self.source}:{self.heading_path} / {self.label}"
        surface = "display math" if self.kind == "DisplayMath" else "inline math"
        return f"{self.source}:{self.heading_path} / {surface}"

    @property
    def surface(self) -> str:
        return "main" if self.source == "manuscript.md" else "supplement"


@dataclass
class SymbolRecord:
    spec: SymbolSpec
    occurrences: list[MathOccurrence]
    raw_forms: set[str]
    defining_occurrences: list[MathOccurrence]
    verdict: str
    note: str


def literal(label: str, text: str) -> PatternSpec:
    return PatternSpec(label=label, regex=re.escape(text))


def _s(pattern: str) -> str:
    return pattern


SYMBOL_SPECS: tuple[SymbolSpec, ...] = (
    SymbolSpec(
        key="W",
        canonical="W",
        family="physical model",
        meaning="Out-of-plane displacement field of the passive structure.",
        recommended="W",
        patterns=(literal("W(x,y,ω;θ)", r"W(x,y,\omega;\theta)"), literal("W(x_L,y_L,ω;θ)", r"W(x_L,y_L,\omega;\theta)")),
        defining_patterns=(r"\\mathcal L_\\omega\s*W\([^)]*\\omega;\\theta\)\s*=", r"W\([^)]*\\omega;\\theta\)\s*=", r"W\("),
    ),
    SymbolSpec(
        key="mathcal_L_omega",
        canonical=r"\mathcal L_\omega",
        family="physical model",
        meaning="Frequency-domain structural operator.",
        recommended=r"\mathcal L_\omega",
        patterns=(literal(r"\mathcal L_\omega", r"\mathcal L_\omega"),),
        defining_patterns=(r"\\mathcal L_\\omega\s+W", r"\\mathcal L_\\omega\s*="),
    ),
    SymbolSpec(
        key="P",
        canonical="P",
        family="physical model",
        meaning="Effective distributed loading induced by incident sound.",
        recommended="P",
        patterns=(literal("P", r"P("),),
        defining_patterns=(r"P\(\s*\\cdot,\s*\\cdot,\s*\\omega;\\theta\)",),
    ),
    SymbolSpec(
        key="D_p",
        canonical=r"D_p",
        family="physical model",
        meaning="Plate bending stiffness in the representative operator.",
        recommended=r"D_p",
        patterns=(literal(r"D_p", r"D_p"),),
        defining_patterns=(r"D_p\\nabla\^4",),
    ),
    SymbolSpec(
        key="rho_t_areal",
        canonical=r"\rho t",
        family="physical model",
        meaning="Areal mass density term in the plate operator.",
        recommended=r"\rho t",
        patterns=(literal(r"\rho t", r"\rho t"),),
        defining_patterns=(r"\\rho t\\,\\omega\^2",),
    ),
    SymbolSpec(
        key="c_d",
        canonical=r"c_d",
        family="physical model",
        meaning="Effective damping term in the plate operator.",
        recommended=r"c_d",
        patterns=(literal(r"c_d", r"c_d"),),
        defining_patterns=(r"i\\omega c_d",),
    ),
    SymbolSpec(
        key="Y",
        canonical="Y",
        family="physical model",
        meaning="Single-point velocity response as a function of frequency and direction.",
        recommended="Y",
        patterns=(literal("Y(ω;θ)", r"Y(\omega;\theta)"), literal("Y(ω_k;θ_e)", r"Y(\omega_k;\theta_e)")),
        defining_patterns=(r"Y\(\\omega;\\theta\)\s*=",),
    ),
    SymbolSpec(
        key="V_phys",
        canonical=r"V(x_L,y_L,\omega;\theta)",
        family="physical model",
        meaning="Physical single-point velocity response at the LDV location.",
        recommended=r"V(x_L,y_L,\omega;\theta)",
        patterns=(literal(r"V(x_L,y_L,\omega;\theta)", r"V(x_L,y_L,\omega;\theta)"), literal(r"V(x_L,y_L,\omega)", r"V(x_L,y_L,\omega)")),
        defining_patterns=(r"V\(x_L,y_L,\\omega;\\theta\)\s*=", r"V\(x_L,y_L,\\omega\)\s*="),
        note="Kept separate from STFT coefficient notation.",
    ),
    SymbolSpec(
        key="G_omega",
        canonical=r"G_\omega",
        family="physical model",
        meaning="Green's function of the structural operator.",
        recommended=r"G_\omega",
        patterns=(literal(r"G_\omega", r"G_\omega"),),
        defining_patterns=(r"G_\\omega",),
    ),
    SymbolSpec(
        key="Omega",
        canonical=r"\Omega",
        family="physical model",
        meaning="Spatial integration domain of the passive structure.",
        recommended=r"\Omega",
        patterns=(literal(r"\Omega", r"\Omega"),),
        defining_patterns=(r"\\iint_\\Omega",),
    ),
    SymbolSpec(
        key="R",
        canonical="R",
        family="physical model",
        meaning="Number of appreciable structural modes in the analysis band.",
        recommended="R",
        patterns=(PatternSpec("R", r"(?<![A-Za-z\\])R(?![A-Za-z])"),),
        defining_patterns=(r"\^\{R\}", r"\\lesssim R"),
    ),
    SymbolSpec(
        key="s_m",
        canonical=r"s_m(\omega)",
        family="physical model",
        meaning="Modal spectral fingerprint.",
        recommended=r"s_m(\omega)",
        patterns=(literal(r"s_m(\omega)", r"s_m(\omega)"),),
        defining_patterns=(r"\}_\{s_m\(\\omega\)\}", r"s_m\(\\omega\)"),
    ),
    SymbolSpec(
        key="alpha_m",
        canonical=r"\alpha_m(\theta)",
        family="physical model",
        meaning="Direction-dependent modal participation weight.",
        recommended=r"\alpha_m(\theta)",
        patterns=(
            literal(r"\alpha_m(\theta,\omega)", r"\alpha_m(\theta,\omega)"),
            literal(r"\alpha_m(\theta)", r"\alpha_m(\theta)"),
            literal(r"\alpha_m(\theta_e,\omega_k)", r"\alpha_m(\theta_e,\omega_k)"),
        ),
        defining_patterns=(r"\\alpha_m\(\\theta,\\omega\)",),
    ),
    SymbolSpec(
        key="phi_m",
        canonical=r"\phi_m",
        family="physical model",
        meaning="Structural mode shape.",
        recommended=r"\phi_m",
        patterns=(literal(r"\phi_m", r"\phi_m"),),
        defining_patterns=(r"\\phi_m\(x_L,y_L\)",),
    ),
    SymbolSpec(
        key="m_m",
        canonical=r"m_m",
        family="physical model",
        meaning="Modal mass.",
        recommended=r"m_m",
        patterns=(literal(r"m_m", r"m_m"),),
        defining_patterns=(r"m_m\\left",),
    ),
    SymbolSpec(
        key="omega_m",
        canonical=r"\omega_m",
        family="physical model",
        meaning="Modal resonance frequency.",
        recommended=r"\omega_m",
        patterns=(literal(r"\omega_m", r"\omega_m"),),
        defining_patterns=(r"\\omega_m\^2",),
    ),
    SymbolSpec(
        key="zeta_m",
        canonical=r"\zeta_m",
        family="physical model",
        meaning="Modal damping ratio.",
        recommended=r"\zeta_m",
        patterns=(literal(r"\zeta_m", r"\zeta_m"),),
        defining_patterns=(r"\\zeta_m\\omega_m\\omega",),
    ),
    SymbolSpec(
        key="mathcal_H",
        canonical=r"\mathcal H",
        family="physical model",
        meaning="Ideal sampled complex transfer matrix before measurement and nonlinear preprocessing.",
        recommended=r"\mathcal H",
        patterns=(literal(r"\mathcal H_{k,e}", r"\mathcal H_{k,e}"), literal(r"\mathcal H", r"\mathcal H")),
        defining_patterns=(r"\\mathcal H_\{k,e\}\s*=", r"\\mathcal H\s*\\approx"),
    ),
    SymbolSpec(
        key="S_factor",
        canonical="S",
        family="physical model",
        meaning="Shared spectral basis in the approximate factorization of the ideal transfer matrix.",
        recommended="S",
        patterns=(literal("S", r"S_{k,m}"),),
        defining_patterns=(r"S_\{k,m\}\s*=",),
    ),
    SymbolSpec(
        key="B_factor",
        canonical="B",
        family="physical model",
        meaning="Directional weight matrix in the approximate factorization of the ideal transfer matrix.",
        recommended="B",
        patterns=(literal("B", r"B_{e,m}"),),
        defining_patterns=(r"B_\{e,m\}\s*=",),
    ),
    SymbolSpec(
        key="v_en",
        canonical=r"v_{e,n}(t)",
        family="measured fingerprints and calibration",
        meaning="Measured LDV waveform at angle index e and trial index n.",
        recommended=r"v_{e,n}(t)",
        patterns=(literal(r"v_{e,n}(t)", r"v_{e,n}(t)"),),
        defining_patterns=(r"v_\{e,n\}\(t\)",),
    ),
    SymbolSpec(
        key="V_stft",
        canonical=r"V[k,t]",
        family="measured fingerprints and calibration",
        meaning="Complex STFT coefficient of the measured waveform.",
        recommended=r"V[k,t]",
        patterns=(literal(r"V_{e,n}[k,t]", r"V_{e,n}[k,t]"), literal(r"V[k,t]", r"V[k,t]")),
        defining_patterns=(r"V_\{e,n\}\[k,t\]\s*=", r"V\[k,t\]\s*\|?\^?2"),
        note="Main text uses a clip-local shorthand; Supplementary Methods 2 restores explicit angle and trial indices.",
    ),
    SymbolSpec(
        key="Shat",
        canonical=r"\widehat S",
        family="measured fingerprints and calibration",
        meaning="Time-averaged power spectrum statistic.",
        recommended=r"\widehat S",
        patterns=(literal(r"\widehat{S}_{e,n}[k]", r"\widehat{S}_{e,n}[k]"), literal(r"\widehat{S}(\omega_k;\theta)", r"\widehat{S}(\omega_k;\theta)")),
        defining_patterns=(r"\\widehat\{S\}_\{e,n\}\[k\]\s*=", r"\\widehat\{S\}\(\\omega_k;\\theta\)\s*="),
        note="Indexed and unindexed forms refer to the same power-spectrum statistic at different descriptive levels.",
    ),
    SymbolSpec(
        key="y",
        canonical="y",
        family="measured fingerprints and calibration",
        meaning="Log-power fingerprint before standardization.",
        recommended="y",
        patterns=(literal(r"y_{e,n}[k]", r"y_{e,n}[k]"), literal(r"y[k]", r"y[k]")),
        defining_patterns=(r"y_\{e,n\}\[k\]\s*=", r"y\[k\]\s*="),
        note="Main text uses suppressed clip indices; supplement keeps explicit calibration indices.",
    ),
    SymbolSpec(
        key="mu",
        canonical=r"\mu",
        family="measured fingerprints and calibration",
        meaning="Per-frequency calibration mean.",
        recommended=r"\mu",
        patterns=(literal(r"\mu[k]", r"\mu[k]"),),
        defining_patterns=(r"\\mu\[k\]\s*=",),
    ),
    SymbolSpec(
        key="sigma",
        canonical=r"\sigma",
        family="measured fingerprints and calibration",
        meaning="Per-frequency calibration scale.",
        recommended=r"\sigma",
        patterns=(literal(r"\sigma[k]", r"\sigma[k]"),),
        defining_patterns=(r"\\sigma\[k\]\s*=",),
    ),
    SymbolSpec(
        key="epsilon",
        canonical=r"\epsilon",
        family="measured fingerprints and calibration",
        meaning="Small positive stabilizer in log-power features.",
        recommended=r"\epsilon",
        patterns=(literal(r"\epsilon", r"\epsilon"),),
        defining_patterns=(),
    ),
    SymbolSpec(
        key="epsilon_sigma",
        canonical=r"\epsilon_\sigma",
        family="measured fingerprints and calibration",
        meaning="Small positive stabilizer in calibration-variance normalization.",
        recommended=r"\epsilon_\sigma",
        patterns=(literal(r"\epsilon_\sigma", r"\epsilon_\sigma"),),
        defining_patterns=(),
    ),
    SymbolSpec(
        key="tilde_y",
        canonical=r"\tilde y",
        family="measured fingerprints and calibration",
        meaning="Standardized measured fingerprint in the full feature space.",
        recommended=r"\tilde y",
        patterns=(literal(r"\tilde y_{e,n}[k]", r"\tilde y_{e,n}[k]"), literal(r"\tilde y[k]", r"\tilde y[k]"), literal(r"\tilde y", r"\tilde y")),
        defining_patterns=(r"\\tilde y_\{e,n\}\[k\]\s*=",),
    ),
    SymbolSpec(
        key="h_e",
        canonical=r"h_e",
        family="measured fingerprints and calibration",
        meaning="Angle-indexed empirical prototype fingerprint.",
        recommended=r"h_e",
        patterns=(literal(r"h_e", r"h_e"),),
        defining_patterns=(r"h_e\s*=",),
    ),
    SymbolSpec(
        key="H",
        canonical="H",
        family="measured fingerprints and calibration",
        meaning="Empirical dictionary of standardized measured fingerprints.",
        recommended="H",
        patterns=(
            PatternSpec("H", r"(?<!\\mathcal )(?<![A-Za-z\\])H(?![A-Za-z_])"),
            PatternSpec("H[k,e]", r"H\[k,e\]"),
            PatternSpec("Hx", r"Hx"),
        ),
        defining_patterns=(r"H=\[h_1,\\dots,h_E\]", r"A = U_r\^\\top H"),
    ),
    SymbolSpec(
        key="H_fig",
        canonical=r"H_{\mathrm{fig}}",
        family="measured fingerprints and calibration",
        meaning="Centered-magnitude fingerprint matrix used for compactness and local-order analysis.",
        recommended=r"H_{\mathrm{fig}}",
        patterns=(literal(r"H_{\mathrm{fig}}", r"H_{\mathrm{fig}}"),),
        defining_patterns=(r"H_\{\\mathrm\{fig\}\}\[k,e\]\s*=", r"H_\{\\mathrm\{fig\}\}\s*= U\\Sigma V"),
    ),
    SymbolSpec(
        key="U",
        canonical="U",
        family="measured fingerprints and calibration",
        meaning="Left singular-vector matrix of the centered-magnitude representation.",
        recommended="U",
        patterns=(literal("U", r"U\Sigma"), literal("U_r", r"U_r^\top")),
        defining_patterns=(r"H_\{\\mathrm\{fig\}\}\s*= U\\Sigma V\^\\top", r"z = U_r\^\\top"),
        note="Includes retained subspace U_r used in the reduced surrogate.",
    ),
    SymbolSpec(
        key="Sigma",
        canonical=r"\Sigma",
        family="measured fingerprints and calibration",
        meaning="Singular-value matrix of the centered-magnitude representation.",
        recommended=r"\Sigma",
        patterns=(literal(r"\Sigma", r"\Sigma"),),
        defining_patterns=(r"H_\{\\mathrm\{fig\}\}\s*= U\\Sigma V\^\\top",),
    ),
    SymbolSpec(
        key="x",
        canonical="x",
        family="reduced surrogate",
        meaning="Direction-level surrogate coefficient vector.",
        recommended="x",
        patterns=(
            PatternSpec(r"\|x\|_0", r"\\\|x\\\|_0"),
            PatternSpec("Hx", r"Hx"),
            PatternSpec("A x", r"A(?:\\,|\s)*x"),
            literal(r"x_{\mathcal S_t}^{(t)}", r"x_{\mathcal S_t}^{(t)}"),
            PatternSpec(r"x_e^{(K)}", r"x_e\^\{\(K\)\}"),
        ),
        defining_patterns=(r"\\tilde y\s*\\approx\s*Hx", r"z\s*\\approx\s*A\s*x"),
        note="Direction-level coefficient family; grouped state x_t is tracked separately.",
    ),
    SymbolSpec(
        key="K",
        canonical="K",
        family="reduced surrogate",
        meaning="Residual-correction budget / pursuit depth.",
        recommended="K",
        patterns=(PatternSpec("K", r"(?<![A-Za-z\\])K(?![A-Za-z_])"),),
        defining_patterns=(r"\\\|x\\\|_0\s*\\le K",),
        note="K_sup is tracked separately under grouped routing.",
    ),
    SymbolSpec(
        key="z",
        canonical="z",
        family="reduced surrogate",
        meaning="Reduced-order fingerprint in the retained singular subspace.",
        recommended="z",
        patterns=(PatternSpec("z", r"(?<![A-Za-z\\])z(?![A-Za-z])"),),
        defining_patterns=(r"z = U_r\^\\top \\tilde y",),
    ),
    SymbolSpec(
        key="A",
        canonical="A",
        family="reduced surrogate",
        meaning="Reduced template matrix in the retained singular subspace.",
        recommended="A",
        patterns=(PatternSpec("A", r"(?<![A-Za-z\\])A(?![A-Za-z])"), literal(r"A_{\mathcal S_t}", r"A_{\mathcal S_t}")),
        defining_patterns=(r"A = U_r\^\\top H",),
    ),
    SymbolSpec(
        key="rho_t_residual",
        canonical=r"\rho_t",
        family="reduced surrogate",
        meaning="Reduced-space residual in the hard-OMP baseline.",
        recommended=r"\rho_t",
        patterns=(literal(r"\rho_0", r"\rho_0"), literal(r"\rho_t", r"\rho_t")),
        defining_patterns=(r"\\rho_0\s*= z",),
    ),
    SymbolSpec(
        key="S_t",
        canonical=r"\mathcal S_t",
        family="reduced surrogate",
        meaning="Active support set in the hard-OMP baseline.",
        recommended=r"\mathcal S_t",
        patterns=(literal(r"\mathcal S_0", r"\mathcal S_0"), literal(r"\mathcal S_t", r"\mathcal S_t")),
        defining_patterns=(r"\\mathcal S_0\s*= \\varnothing",),
    ),
    SymbolSpec(
        key="c_e",
        canonical=r"c_e",
        family="reduced surrogate",
        meaning="Hard-OMP correlation score on reduced templates.",
        recommended=r"c_e",
        patterns=(literal(r"c_e(\rho)", r"c_e(\rho)"),),
        defining_patterns=(r"c_e\(\\rho\)\s*=",),
    ),
    SymbolSpec(
        key="j_t",
        canonical=r"j_t",
        family="reduced surrogate",
        meaning="Greedy support index selected by hard OMP.",
        recommended=r"j_t",
        patterns=(literal(r"j_t", r"j_t"),),
        defining_patterns=(r"j_t\s*= \\arg\\max",),
    ),
    SymbolSpec(
        key="D",
        canonical="D",
        family="grouped routing / solver",
        meaning="Grouped full-space readout dictionary.",
        recommended="D",
        patterns=(PatternSpec("D", r"(?<![A-Za-z\\])D(?![A-Za-z_])"), literal(r"D=[d_{e,m}]", r"D=[d_{e,m}]")),
        defining_patterns=(r"D=\[d_\{e,m\}\]",),
    ),
    SymbolSpec(
        key="d_em",
        canonical=r"d_{e,m}",
        family="grouped routing / solver",
        meaning="Within-direction atom in the grouped readout dictionary.",
        recommended=r"d_{e,m}",
        patterns=(literal(r"d_{e,m}", r"d_{e,m}"),),
        defining_patterns=(r"D=\[d_\{e,m\}\]",),
    ),
    SymbolSpec(
        key="g_0",
        canonical=r"g_0",
        family="grouped routing / solver",
        meaning="Ungated stage-0 grouped physical match.",
        recommended=r"g_0",
        patterns=(literal(r"g_0", r"g_0"),),
        defining_patterns=(r"g_0 = D\^\\top \\tilde y",),
    ),
    SymbolSpec(
        key="g0_grp",
        canonical=r"g_0^{(\mathrm{grp})}",
        family="grouped routing / solver",
        meaning="Direction-level ungated group summary derived from the stage-0 grouped match.",
        recommended=r"g_0^{(\mathrm{grp})}",
        patterns=(literal(r"g_0^{(\mathrm{grp})}[e]", r"g_0^{(\mathrm{grp})}[e]"),),
        defining_patterns=(r"g_0\^\{\(\\mathrm\{grp\}\)\}\[e\]\s*=",),
    ),
    SymbolSpec(
        key="x_t",
        canonical=r"x_t",
        family="grouped routing / solver",
        meaning="Grouped coefficient state on the readout dictionary D.",
        recommended=r"x_t",
        patterns=(literal(r"x_t", r"x_t"), literal(r"x_0", r"x_0")),
        defining_patterns=(r"x_0=0",),
    ),
    SymbolSpec(
        key="r_t",
        canonical=r"r_t",
        family="grouped routing / solver",
        meaning="Full-space residual in the routed update recursion.",
        recommended=r"r_t",
        patterns=(literal(r"r_0", r"r_0"), literal(r"r_t", r"r_t"), literal(r"r_{t+1}", r"r_{t+1}")),
        defining_patterns=(r"r_0=\\tilde y",),
    ),
    SymbolSpec(
        key="g_t",
        canonical=r"g_t",
        family="grouped routing / solver",
        meaning="Grouped physical match between the current residual and the dictionary atoms.",
        recommended=r"g_t",
        patterns=(literal(r"g_t", r"g_t"),),
        defining_patterns=(r"g_t = D\^\\top r_t",),
    ),
    SymbolSpec(
        key="q_t",
        canonical=r"q_t",
        family="grouped routing / solver",
        meaning="Learned query derived from the current residual for routing.",
        recommended=r"q_t",
        patterns=(literal(r"q_t", r"q_t"),),
        defining_patterns=(),
    ),
    SymbolSpec(
        key="k_em",
        canonical=r"k_e",
        family="grouped routing / solver",
        meaning="Learned routing key associated with grouped atom (e,m).",
        recommended=r"k_e",
        patterns=(literal(r"k_{e,m}", r"k_{e,m}"), literal(r"k_e", r"k_e")),
        defining_patterns=(),
        note="Main text uses direction-level shorthand; Supplementary Methods 4 restores the atom index for the grouped construction.",
    ),
    SymbolSpec(
        key="d_k",
        canonical=r"d_k",
        family="grouped routing / solver",
        meaning="Key dimension used in scaled dot-product routing scores.",
        recommended=r"d_k",
        patterns=(literal(r"d_k", r"d_k"),),
        defining_patterns=(),
    ),
    SymbolSpec(
        key="s_atom",
        canonical=r"s_t^{(\mathrm{atom})}",
        family="grouped routing / solver",
        meaning="Atom-level learned routing compatibility score.",
        recommended=r"s_t^{(\mathrm{atom})}",
        patterns=(literal(r"s_t^{(\mathrm{atom})}[e,m]", r"s_t^{(\mathrm{atom})}[e,m]"),),
        defining_patterns=(r"s_t\^\{\(\\mathrm\{atom\}\)\}\[e,m\]\s*=",),
    ),
    SymbolSpec(
        key="s_exp",
        canonical=r"s_t[e]",
        family="grouped routing / solver",
        meaning="Direction-level expert / routing score used for grouped readout.",
        recommended=r"s_t[e]",
        patterns=(literal(r"s_t[e]", r"s_t[e]"), literal(r"s_t^{(\mathrm{exp})}[e]", r"s_t^{(\mathrm{exp})}[e]")),
        defining_patterns=(r"s_t\[e\]\s*=", r"s_t\^\{\(\\mathrm\{exp\}\)\}\[e\]\s*="),
        drift_on_multiple_forms=True,
        note="Canonical shared direction-level score; any remaining s_t^{(\\mathrm{exp})}[e] usage should be treated as drift.",
    ),
    SymbolSpec(
        key="w_t",
        canonical=r"w_t",
        family="grouped routing / solver",
        meaning="Direction-level routing gate over expert groups.",
        recommended=r"w_t",
        patterns=(literal(r"w_t", r"w_t"),),
        defining_patterns=(r"w_t = \\mathrm\{GumbelSoftmax\}",),
    ),
    SymbolSpec(
        key="u_t",
        canonical=r"u_t^{(e)}",
        family="grouped routing / solver",
        meaning="Within-group atom-level routing gate.",
        recommended=r"u_t^{(e)}",
        patterns=(literal(r"u_t^{(e)}", r"u_t^{(e)}"),),
        defining_patterns=(r"u_t\^\{\(e\)\}\s*=",),
    ),
    SymbolSpec(
        key="W_t",
        canonical=r"W_t",
        family="grouped routing / solver",
        meaning="Combined direction-level and atom-level routing gate.",
        recommended=r"W_t",
        patterns=(literal(r"W_t[e,m]", r"W_t[e,m]"),),
        defining_patterns=(r"W_t\[e,m\]\s*=",),
    ),
    SymbolSpec(
        key="Delta_x_t",
        canonical=r"\Delta x_t",
        family="grouped routing / solver",
        meaning="Stagewise gated coefficient update.",
        recommended=r"\Delta x_t",
        patterns=(literal(r"\Delta x_t[e,m]", r"\Delta x_t[e,m]"), literal(r"\Delta x_t", r"\Delta x_t")),
        defining_patterns=(r"\\Delta x_t\[e,m\]\s*=", r"\\Delta x_t ="),
    ),
    SymbolSpec(
        key="bar_s",
        canonical=r"\bar s[e]",
        family="grouped routing / solver",
        meaning="Readout score aggregated across supervised stages.",
        recommended=r"\bar s[e]",
        patterns=(literal(r"\bar s[e]", r"\bar s[e]"),),
        defining_patterns=(r"\\bar s\[e\]\s*=",),
    ),
    SymbolSpec(
        key="hat_theta",
        canonical=r"\hat\theta",
        family="grouped routing / solver",
        meaning="Predicted direction after readout.",
        recommended=r"\hat\theta",
        patterns=(literal(r"\hat\theta", r"\hat\theta"),),
        defining_patterns=(r"\\hat\\theta\s*=",),
    ),
    SymbolSpec(
        key="K_sup",
        canonical=r"K_{\mathrm{sup}}",
        family="grouped routing / solver",
        meaning="Number of supervised stages used in the readout aggregation.",
        recommended=r"K_{\mathrm{sup}}",
        patterns=(literal(r"K_{\mathrm{sup}}", r"K_{\mathrm{sup}}"),),
        defining_patterns=(r"K_\{\\mathrm\{sup\}\}",),
    ),
    SymbolSpec(
        key="mathcal_L",
        canonical=r"\mathcal L",
        family="grouped routing / solver",
        meaning="Composite training objective for the routed solver.",
        recommended=r"\mathcal L",
        patterns=(literal(r"\mathcal L", r"\mathcal L ="),),
        defining_patterns=(r"\\mathcal L = ",),
    ),
    SymbolSpec(
        key="L_rec",
        canonical=r"\mathcal L_{\mathrm{rec}}",
        family="grouped routing / solver",
        meaning="Reconstruction term in the routed solver objective.",
        recommended=r"\mathcal L_{\mathrm{rec}}",
        patterns=(literal(r"\mathcal L_{\mathrm{rec}}", r"\mathcal L_{\mathrm{rec}}"),),
        defining_patterns=(r"\\mathcal L_\{\\mathrm\{rec\}\}\s*=",),
    ),
    SymbolSpec(
        key="L_mono",
        canonical=r"\mathcal L_{\mathrm{mono}}",
        family="grouped routing / solver",
        meaning="Monotonicity regularizer on stagewise residual descent.",
        recommended=r"\mathcal L_{\mathrm{mono}}",
        patterns=(literal(r"\mathcal L_{\mathrm{mono}}", r"\mathcal L_{\mathrm{mono}}"),),
        defining_patterns=(r"\\mathcal L_\{\\mathrm\{mono\}\}\s*=",),
    ),
    SymbolSpec(
        key="L_cls",
        canonical=r"\mathcal L_{\mathrm{cls}}",
        family="grouped routing / solver",
        meaning="Classification term in the routed solver objective.",
        recommended=r"\mathcal L_{\mathrm{cls}}",
        patterns=(literal(r"\mathcal L_{\mathrm{cls}}", r"\mathcal L_{\mathrm{cls}}"),),
        defining_patterns=(r"\\mathcal L_\{\\mathrm\{cls\}\}\s*=",),
    ),
    SymbolSpec(
        key="loss_weights",
        canonical=r"\alpha,\beta,\gamma",
        family="grouped routing / solver",
        meaning="Loss weights for reconstruction, monotonicity, and classification terms.",
        recommended=r"\alpha,\beta,\gamma",
        patterns=(literal(r"\alpha", r"\alpha"), literal(r"\beta", r"\beta"), literal(r"\gamma", r"\gamma")),
        defining_patterns=(r"\(\\alpha,\\beta,\\gamma\)=",),
    ),
    SymbolSpec(
        key="e_star",
        canonical=r"e^\star",
        family="grouped routing / solver",
        meaning="Ground-truth direction-group index in the classification loss.",
        recommended=r"e^\star",
        patterns=(literal(r"e^\star", r"e^\star"),),
        defining_patterns=(),
    ),
    SymbolSpec(
        key="rho_uv",
        canonical=r"\rho(u,v)",
        family="statistics",
        meaning="Pearson correlation coefficient.",
        recommended=r"\rho(u,v)",
        patterns=(literal(r"\rho(u,v)", r"\rho(u,v)"), literal(r"\rho(h_e,h_{e'})", r"\rho(h_e,h_{e'})")),
        defining_patterns=(r"\\rho\(u,v\)\s*=",),
    ),
    SymbolSpec(
        key="W_e_set",
        canonical=r"\mathcal W_e",
        family="statistics",
        meaning="Within-angle correlation set.",
        recommended=r"\mathcal W_e",
        patterns=(literal(r"\mathcal W_e", r"\mathcal W_e"),),
        defining_patterns=(r"\\mathcal W_e\s*=",),
    ),
    SymbolSpec(
        key="r_within",
        canonical=r"\bar r_{\mathrm{within}}",
        family="statistics",
        meaning="Mean within-angle correlation.",
        recommended=r"\bar r_{\mathrm{within}}",
        patterns=(literal(r"\bar r_{\mathrm{within}}(e)", r"\bar r_{\mathrm{within}}(e)"), literal(r"\bar{r}", r"\bar{r}")),
        defining_patterns=(r"\\bar r_\{\\mathrm\{within\}\}\(e\)\s*=",),
        note="Main text also uses the shorter summary notation \\bar r in prose-level statistics.",
    ),
    SymbolSpec(
        key="B_e_set",
        canonical=r"\mathcal B_e",
        family="statistics",
        meaning="Between-angle correlation set anchored at angle e.",
        recommended=r"\mathcal B_e",
        patterns=(literal(r"\mathcal B_e", r"\mathcal B_e"),),
        defining_patterns=(r"\\mathcal B_e\s*=",),
    ),
    SymbolSpec(
        key="r_between",
        canonical=r"\bar r_{\mathrm{between}}",
        family="statistics",
        meaning="Mean between-angle correlation.",
        recommended=r"\bar r_{\mathrm{between}}",
        patterns=(literal(r"\bar r_{\mathrm{between}}(e)", r"\bar r_{\mathrm{between}}(e)"),),
        defining_patterns=(r"\\bar r_\{\\mathrm\{between\}\}\(e\)\s*=",),
    ),
    SymbolSpec(
        key="Delta_r",
        canonical=r"\Delta r",
        family="statistics",
        meaning="Per-angle discriminability margin.",
        recommended=r"\Delta r",
        patterns=(literal(r"\Delta r(e)", r"\Delta r(e)"), literal(r"\Delta \bar r", r"\Delta \bar r")),
        defining_patterns=(r"\\Delta r\(e\)\s*=",),
        note="Main text uses prose-level summary notation \\Delta \\bar r.",
    ),
    SymbolSpec(
        key="cohens_d",
        canonical="d",
        family="statistics",
        meaning="Cohen's d effect size or reported scalar effect-size summary.",
        recommended="d",
        patterns=(literal("d", r"d ="),),
        defining_patterns=(r"d\s*=",),
    ),
    SymbolSpec(
        key="S_corr",
        canonical=r"S_{e,e'}",
        family="statistics",
        meaning="Inter-angle prototype similarity matrix.",
        recommended=r"S_{e,e'}",
        patterns=(literal(r"S_{e,e'}", r"S_{e,e'}"),),
        defining_patterns=(r"S_\{e,e'\}\s*=",),
    ),
    SymbolSpec(
        key="r_corr",
        canonical="r",
        family="statistics",
        meaning="Reported scalar correlation summary, including structure-alignment metrics.",
        recommended="r",
        patterns=(literal("r", r"r ="),),
        defining_patterns=(r"r\s*=",),
        note="Used in prose-level correlation summaries rather than as a standalone model variable.",
    ),
    SymbolSpec(
        key="abs_H",
        canonical=r"|H|",
        family="cross-object descriptors",
        meaning="Magnitude template matrix used in centered-magnitude and cross-object descriptors.",
        recommended=r"|H|",
        patterns=(literal(r"|H|", r"|H|"),),
        defining_patterns=(),
    ),
)


GENERIC_TOKEN_RE = re.compile(
    r"""
    \\mathcal\{[^}]+\}(?:_\{[^}]+\}|_[A-Za-z0-9]+)?|
    \\(?:widehat|tilde|hat|bar|Delta)\s*[A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9]+)?(?:\^\{[^}]+\})?(?:\[[^\]]+\])?|
    \\[A-Za-z]+(?:_\{[^}]+\}|_[A-Za-z0-9]+)?(?:\^\{[^}]+\})?(?:\[[^\]]+\])?|
    [A-Za-z](?:_\{[^}]+\}|_[A-Za-z0-9]+)?(?:\^\{[^}]+\})?(?:\[[^\]]+\])?
    """,
    re.X,
)
GENERIC_STOPLIST = {
    r"\sum",
    r"\sqrt",
    r"\frac",
    r"\left",
    r"\right",
    r"\big",
    r"\bigg",
    r"\quad",
    r"\qquad",
    r"\arg",
    r"\argmax",
    r"\max",
    r"\min",
    r"\approx",
    r"\le",
    r"\ge",
    r"\in",
    r"\top",
    r"\cdot",
    r"\dots",
    r"\lVert",
    r"\rVert",
    r"\langle",
    r"\rangle",
    r"\text",
    r"\tag",
    "subject",
    "to",
}


def stringify_inlines(inlines: list[dict]) -> str:
    parts: list[str] = []
    for item in inlines:
        tag = item["t"]
        if tag == "Str":
            parts.append(item["c"])
        elif tag == "Space":
            parts.append(" ")
        elif tag == "Code":
            parts.append(item["c"][1])
        elif tag == "Math":
            parts.append(item["c"][1])
    return "".join(parts).strip()


def run_pandoc(path: Path) -> dict:
    proc = subprocess.run(
        ["pandoc", "-f", "markdown+tex_math_single_backslash", "-t", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def strip_tag(payload: str) -> str:
    return re.sub(r"\\tag\{[^}]+\}", "", payload).strip()


def extract_equation_label(payload: str) -> str | None:
    tag_match = re.search(r"\\tag\{([^}]+)\}", payload)
    if tag_match:
        return f"Eq. ({tag_match.group(1)})"
    trimmed = payload.strip()
    num_match = re.search(r"\((\d+)\)\s*$", trimmed)
    if num_match:
        return f"Eq. ({num_match.group(1)})"
    return None


def extract_occurrences(path: Path) -> list[MathOccurrence]:
    ast = run_pandoc(path)
    occurrences: list[MathOccurrence] = []
    heading_stack: list[tuple[int, str]] = []
    order = 0

    def current_heading_path() -> str:
        headings = [text for level, text in heading_stack if level >= 2]
        return " / ".join(headings) if headings else "(preamble)"

    def walk(node):
        nonlocal order
        if isinstance(node, dict):
            if node.get("t") == "Math":
                kind = node["c"][0]["t"]
                raw = node["c"][1]
                occurrences.append(
                    MathOccurrence(
                        source=path.name,
                        heading_path=current_heading_path(),
                        kind=kind,
                        raw_tex=raw,
                        label=extract_equation_label(raw),
                        order=order,
                    )
                )
                order += 1
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for block in ast["blocks"]:
        if block["t"] == "Header":
            level = block["c"][0]
            text = stringify_inlines(block["c"][2])
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
        walk(block)

    return occurrences


def summarize_occurrences(occurrences: Iterable[MathOccurrence], surface: str) -> str:
    selected = [occ for occ in occurrences if occ.surface == surface]
    if not selected:
        return "—"
    first = selected[0]
    labels = []
    seen = set()
    for occ in selected:
        lab = occ.label or ("display math" if occ.kind == "DisplayMath" else "inline math")
        if lab not in seen:
            seen.add(lab)
            labels.append(lab)
        if len(labels) == 3:
            break
    return f"{len(selected)} occurrence(s); first: {first.heading_path}; refs: {', '.join(labels)}"


def find_matching_forms(spec: SymbolSpec, payload: str) -> list[tuple[str, str]]:
    forms: list[tuple[str, str]] = []
    for pattern in spec.patterns:
        for match in re.finditer(pattern.regex, payload):
            forms.append((pattern.label, match.group(0)))
    return forms


def has_definition(spec: SymbolSpec, payload: str) -> bool:
    if not spec.defining_patterns:
        return False
    return any(re.search(pattern, payload) for pattern in spec.defining_patterns)


def collect_records(occurrences: list[MathOccurrence]) -> list[SymbolRecord]:
    matched_by_spec: dict[str, list[MathOccurrence]] = defaultdict(list)
    forms_by_spec: dict[str, set[str]] = defaultdict(set)
    defs_by_spec: dict[str, list[MathOccurrence]] = defaultdict(list)
    actual_form_to_specs: dict[str, set[str]] = defaultdict(set)

    for occ in occurrences:
        for spec in SYMBOL_SPECS:
            matches = find_matching_forms(spec, occ.raw_tex)
            if not matches:
                continue
            matched_by_spec[spec.key].append(occ)
            forms_by_spec[spec.key].update(label for label, _ in matches)
            if has_definition(spec, occ.raw_tex):
                defs_by_spec[spec.key].append(occ)
            for label, matched_text in matches:
                if label in NON_COLLISION_LABELS:
                    continue
                actual_form_to_specs[matched_text].add(spec.key)

    collided_specs = {
        spec_id
        for spec_ids in actual_form_to_specs.values()
        if len(spec_ids) > 1
        for spec_id in spec_ids
    }

    records: list[SymbolRecord] = []
    for spec in SYMBOL_SPECS:
        occs = matched_by_spec.get(spec.key, [])
        if not occs:
            continue
        defs = defs_by_spec.get(spec.key, [])
        raw_forms = forms_by_spec.get(spec.key, set())
        surfaces = {occ.surface for occ in occs}
        note = spec.note
        if spec.key in collided_specs:
            verdict = "collision"
            if note:
                note = f"{note} Same raw form also matched another canonical entry."
            else:
                note = "Same raw form also matched another canonical entry."
        elif spec.drift_on_multiple_forms and len(raw_forms) > 1:
            verdict = "alias drift"
            drift_note = f"Observed forms: {', '.join(sorted(raw_forms))}."
            note = f"{note} {drift_note}".strip()
        elif spec.defining_patterns and not defs:
            verdict = "undefined"
            if note:
                note = f"{note} No defining occurrence matched the configured definition rules."
            else:
                note = "No defining occurrence matched the configured definition rules."
        elif surfaces == {"main"}:
            verdict = "main-only"
        elif surfaces == {"supplement"}:
            verdict = "supplement-only"
        else:
            verdict = "consistent"
        records.append(
            SymbolRecord(
                spec=spec,
                occurrences=sorted(occs, key=lambda occ: occ.order),
                raw_forms=raw_forms,
                defining_occurrences=sorted(defs, key=lambda occ: occ.order),
                verdict=verdict,
                note=note,
            )
        )
    return sorted(
        records,
        key=lambda rec: (FAMILY_ORDER.index(rec.spec.family), rec.spec.canonical.lower()),
    )


def extract_generic_tokens(payload: str) -> list[str]:
    text = re.sub(r"\\tag\{[^}]+\}", "", payload)
    text = re.sub(r"\\text\{[^}]*\}", "", text)
    tokens = []
    seen = set()
    for match in GENERIC_TOKEN_RE.finditer(text):
        token = match.group(0).strip()
        if not token or token in GENERIC_STOPLIST:
            continue
        if token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    return tokens


def build_unmapped_tokens(occurrences: list[MathOccurrence], records: list[SymbolRecord]) -> dict[str, set[str]]:
    matched_forms = set()
    for record in records:
        matched_forms.update(record.raw_forms)
    unmapped: dict[str, set[str]] = defaultdict(set)
    for occ in occurrences:
        for token in extract_generic_tokens(occ.raw_tex):
            if token in matched_forms:
                continue
            unmapped[occ.source].add(token)
    return unmapped


def format_table(rows: list[list[str]]) -> str:
    header = "| Canonical | Meaning | First defining location | Main text | Supplement | Verdict | Recommended | Note |\n"
    sep = "| --- | --- | --- | --- | --- | --- | --- | --- |\n"
    body = ""
    for row in rows:
        escaped = [cell.replace("|", "\\|").replace("\n", "<br>") for cell in row]
        body += "| " + " | ".join(escaped) + " |\n"
    return header + sep + body


def render_registry(records: list[SymbolRecord]) -> str:
    parts: list[str] = []
    for family in FAMILY_ORDER:
        family_records = [rec for rec in records if rec.spec.family == family]
        if not family_records:
            continue
        parts.append(f"### {family.title()}\n")
        rows = []
        for rec in family_records:
            first_def = rec.defining_occurrences[0].location() if rec.defining_occurrences else rec.occurrences[0].location()
            rows.append(
                [
                    rec.spec.canonical,
                    rec.spec.meaning,
                    first_def,
                    summarize_occurrences(rec.occurrences, "main"),
                    summarize_occurrences(rec.occurrences, "supplement"),
                    rec.verdict,
                    rec.spec.recommended,
                    rec.note or "—",
                ]
            )
        parts.append(format_table(rows))
        parts.append("")
    return "\n".join(parts).strip()


def render_findings(records: list[SymbolRecord]) -> str:
    findings = [rec for rec in records if rec.verdict != "consistent"]
    findings.sort(key=lambda rec: (VERDICT_ORDER[rec.verdict], rec.spec.canonical.lower()))
    if not findings:
        return "No non-consistent findings.\n"
    lines: list[str] = []
    for rec in findings:
        first_loc = rec.defining_occurrences[0].location() if rec.defining_occurrences else rec.occurrences[0].location()
        forms = ", ".join(sorted(rec.raw_forms))
        lines.append(f"- `{rec.spec.canonical}`: `{rec.verdict}`.")
        lines.append(f"  First relevant location: {first_loc}.")
        lines.append(f"  Observed forms: {forms}.")
        lines.append(f"  Recommended canonical form: `{rec.spec.recommended}`.")
        if rec.note:
            lines.append(f"  Note: {rec.note}")
    return "\n".join(lines) + "\n"


def render_appendix(occurrences: list[MathOccurrence], unmapped: dict[str, set[str]]) -> str:
    parts: list[str] = []
    grouped: dict[str, list[MathOccurrence]] = defaultdict(list)
    for occ in occurrences:
        grouped[occ.source].append(occ)
    for source in [target.name for target in TARGETS]:
        parts.append(f"### {source}\n")
        for occ in grouped[source]:
            label = occ.label or ("display math" if occ.kind == "DisplayMath" else "inline math")
            parts.append(f"- `{occ.heading_path}` / `{label}`")
            raw = strip_tag(occ.raw_tex).strip()
            if "\n" in raw:
                parts.append("```tex")
                parts.append(raw)
                parts.append("```")
            else:
                parts.append(f"  - Raw TeX: `{raw}`")
            tokens = extract_generic_tokens(raw)
            parts.append(
                "  - Candidate tokens: "
                + (", ".join(f"`{tok}`" for tok in tokens) if tokens else "—")
            )
        if unmapped.get(source):
            parts.append("")
            parts.append(f"Unmapped candidate tokens in `{source}`: " + ", ".join(f"`{tok}`" for tok in sorted(unmapped[source])))
        parts.append("")
    return "\n".join(parts).strip()


def render_report(occurrences: list[MathOccurrence], records: list[SymbolRecord]) -> str:
    by_source = defaultdict(list)
    for occ in occurrences:
        by_source[occ.source].append(occ)
    unmapped = build_unmapped_tokens(occurrences, records)
    verdict_counts = defaultdict(int)
    for rec in records:
        verdict_counts[rec.verdict] += 1

    overview_lines = [
        "# Math Symbol Consistency Audit",
        "",
        "_Generated by `scripts/paper/audit_math_symbols.py`; do not edit manually._",
        "",
        "## Overview",
        "",
        "Scanned sources:",
    ]
    for path in TARGETS:
        source = path.name
        occs = by_source[source]
        display = sum(1 for occ in occs if occ.kind == "DisplayMath")
        inline = sum(1 for occ in occs if occ.kind == "InlineMath")
        labels = sum(1 for occ in occs if occ.label is not None)
        overview_lines.append(
            f"- `{source}`: {display} display-math node(s), {inline} inline-math node(s), {labels} labeled equation(s)."
        )
    overview_lines.extend(
        [
            "",
            f"Semantic symbol entries audited: {len(records)}.",
            "Overall notation verdict counts:",
        ]
    )
    for verdict in ["collision", "undefined", "alias drift", "main-only", "supplement-only", "consistent"]:
        overview_lines.append(f"- `{verdict}`: {verdict_counts.get(verdict, 0)}")

    report = "\n".join(overview_lines)
    report += "\n\n## Canonical Symbol Registry\n\n"
    report += render_registry(records)
    report += "\n\n## Cross-Surface Findings\n\n"
    report += render_findings(records)
    report += "\n## Coverage Appendix\n\n"
    report += render_appendix(occurrences, unmapped)
    report += "\n"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", type=Path, default=None, help="Write the generated report to this path.")
    parser.add_argument("--stdout", action="store_true", help="Print the generated report to stdout.")
    args = parser.parse_args()

    occurrences: list[MathOccurrence] = []
    for path in TARGETS:
        occurrences.extend(extract_occurrences(path))
    records = collect_records(occurrences)
    report = render_report(occurrences, records)

    wrote = False
    if args.write is not None:
        target = args.write
        if not target.is_absolute():
            target = REPO_ROOT / target
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        wrote = True
    if args.stdout or not wrote:
        try:
            sys.stdout.write(report)
        except BrokenPipeError:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
