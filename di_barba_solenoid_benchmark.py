"""Analytical 10-D multiobjective solenoid benchmark inspired by Di Barba (2016).

Reference:
    P. Di Barba, "Multi-objective wind-driven optimisation and magnet design",
    Electronics Letters 52 (2016), 1216-1218, doi:10.1049/el.2016.1527.

Verified article data: a solenoid of half-length 15 mm, radii 9/10 mm,
control interval +-5 mm, 20 axial sections, 10 independent symmetric
current-density variables, 41 axial observation positions, target B = 0.84 mT.

ASSUMPTIONS / NOT A BIT-EXACT REPRODUCTION:
    * Currents are independent, signed and uniformly distributed in each
      rectangular winding cross section.
    * Adjustable bounds are +-2 MA/m2; the article's bounds are not confirmed.
    * Objectives are normalized mean squared field error and normalized J^2.
      They represent the published field residual / power-loss tradeoff,
      but the source equation images could not be checked for exact factors.
    * Resistivity is used only to report indicative ohmic power in Watts.

The field of each finite annular section is integrated in closed form along
its axis, treating the azimuthal volume current density as uniform.

Design coordinates x_i = J_i / J_SCALE, so x_i are dimensionless and
suitable for ten-dimensional distance calculations in ALSF.

No FEM or pymoo dependency is required to use this model and construct an
independent bounded, convex weighted-sum reference Pareto front.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class SolenoidSettings:
    half_length: float = 15e-3
    inner_radius: float = 9e-3
    outer_radius: float = 10e-3
    control_half_length: float = 5e-3
    n_sections: int = 20
    n_field_samples: int = 41
    target_field: float = 0.84e-3
    j_scale: float = 1e6                # A/m^2: normalized decision variables
    j_min: float = -2e6                # A/m^2: illustrative, not source-verified
    j_max: float = 2e6                 # A/m^2: illustrative, not source-verified
    copper_resistivity: float = 1.724e-8  # ohm*m at ~20 Celsius, illustrative
    independent_j_tolerance: float = 0.1e6  # A/m^2, illustrative


class DiBarbaSolenoid:
    def __init__(self, settings: SolenoidSettings | None = None):
        s = settings or SolenoidSettings()
        if s.n_sections % 2 != 0:
            raise ValueError("Number of sections must be even.")
        if not (0 < s.inner_radius < s.outer_radius):
            raise ValueError("Expected 0 < inner_radius < outer_radius.")
        if not (0 < s.control_half_length <= s.half_length):
            raise ValueError("Control interval must lie within the solenoid.")
        if not (s.j_min < 0 < s.j_max and s.j_scale > 0):
            raise ValueError("Bounds must contain zero; J_SCALE must be positive.")
        self.s = s
        self.n_var = s.n_sections // 2
        self.z_samples = np.linspace(-s.control_half_length,
                                     s.control_half_length, s.n_field_samples)
        self.edges = np.linspace(-s.half_length, s.half_length, s.n_sections + 1)
        self.lower = np.full(self.n_var, s.j_min / s.j_scale)
        self.upper = np.full(self.n_var, s.j_max / s.j_scale)

        # A[m, i] = B at observation m from J_i = 1 A/m^2, both symmetric sections.
        self.A = np.zeros((s.n_field_samples, self.n_var))
        for i in range(self.n_var):
            left = self._section_field_kernel(self.edges[i], self.edges[i + 1])
            j = s.n_sections - 1 - i
            right = self._section_field_kernel(self.edges[j], self.edges[j + 1])
            self.A[:, i] = left + right

        # K maps dimensionless currents to B/B0.
        self.K = self.A * (s.j_scale / s.target_field)
        self.b_target = np.ones(s.n_field_samples)
        self.winding_volume = (np.pi * (s.outer_radius**2 - s.inner_radius**2)
                               * 2 * s.half_length)
        self.p_ref = s.copper_resistivity * self.winding_volume * s.j_scale**2

    @staticmethod
    def _antiderivative(r: float, u: np.ndarray) -> np.ndarray:
        """F(r,u) = u log[r + sqrt(r^2+u^2)], r>0.

        Its difference over two radii equals the radial integral of
        u/sqrt(r^2+u^2). At u=0 this formula is finite and exactly zero.
        """
        return u * np.log(r + np.hypot(r, u))

    def _section_field_kernel(self, z0: float, z1: float,
                              y: np.ndarray | None = None) -> np.ndarray:
        """Axial B(y) in T for a finite annular section, J=1 A/m^2."""
        s = self.s
        if y is None:
            y = self.z_samples
        u1, u0 = z1 - y, z0 - y
        F = self._antiderivative
        return (0.5 * MU0 *
                (F(s.outer_radius, u1) - F(s.inner_radius, u1)
                 - F(s.outer_radius, u0) + F(s.inner_radius, u0)))

    def field(self, x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        """Axial field in T. x = 10 dimensionless independent symmetric J's."""
        x = np.asarray(x, dtype=float).reshape(-1)
        if x.size != self.n_var:
            raise ValueError(f"Expected {self.n_var} decision variables.")
        if z is None:
            return self.A @ (self.s.j_scale * x)
        z = np.asarray(z, dtype=float).reshape(-1)
        result = np.zeros(z.shape, dtype=float)
        for i in range(self.n_var):
            j = self.s.n_sections - 1 - i
            result += (self._section_field_kernel(self.edges[i], self.edges[i + 1], z)
                       + self._section_field_kernel(self.edges[j], self.edges[j + 1], z)) * self.s.j_scale * x[i]
        return result

    def evaluate(self, x: np.ndarray) -> tuple[float, float]:
        """Minimize normalized field MSE and normalized ohmic power proxy."""
        x = np.asarray(x, dtype=float).reshape(self.n_var)
        residual = self.K @ x - self.b_target
        f1 = np.mean(residual**2)
        f2 = np.mean(x**2)
        return float(f1), float(f2)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """2x10 analytical Jacobian of (f1,f2) wrt dimensionless x."""
        x = np.asarray(x, dtype=float).reshape(self.n_var)
        g1 = (2 / self.s.n_field_samples) * self.K.T @ (self.K @ x - self.b_target)
        g2 = (2 / self.n_var) * x
        return np.vstack([g1, g2])

    def ohmic_loss_watt(self, x: np.ndarray) -> float:
        """Illustrative copper loss; all sections assumed fully copper-filled."""
        return self.evaluate(x)[1] * self.p_ref

    def current_density(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) * self.s.j_scale

    def full_winding_j(self, x: np.ndarray) -> np.ndarray:
        """20 signed current densities, symmetric around z=0."""
        j = self.current_density(x)
        return np.concatenate((j, j[::-1]))

    def reference_pareto(self, n_weights: int = 101) -> pd.DataFrame:
        """High-quality Pareto reference by CONVEX bounded least squares.

        Minimize f1 + lambda*f2 where lambda>=0. Both objectives are convex
        quadratic. Weighted solutions are globally optimal for each lambda.
        If published objective norms differ, this reference is valid only
        for this explicitly defined benchmark version.
        """
        lambdas = np.r_[0.0, np.logspace(-8, 7, n_weights)]
        n_obs = self.s.n_field_samples
        K_aug_top = self.K / np.sqrt(n_obs)
        target_top = np.ones(n_obs) / np.sqrt(n_obs)
        eye = np.eye(self.n_var)
        records = []
        for lam in lambdas:
            A_aug = np.vstack([K_aug_top, np.sqrt(lam / self.n_var) * eye])
            b_aug = np.r_[target_top, np.zeros(self.n_var)]
            res = lsq_linear(A_aug, b_aug,
                             bounds=(self.lower, self.upper),
                             tol=1e-11, max_iter=500)
            if not res.success:
                raise RuntimeError(f"Reference solver failed at lambda={lam}: {res.message}")
            x = res.x
            f1, f2 = self.evaluate(x)
            records.append([*x, f1, f2, lam])

        # Zero current is the exactly known minimum-loss endpoint.
        x_zero = np.zeros(self.n_var)
        f1, f2 = self.evaluate(x_zero)
        records.append([*x_zero, f1, f2, np.inf])
        cols = [f"X{i}" for i in range(1, self.n_var+1)] + ["Obj1", "Obj2", "lambda"]
        df = pd.DataFrame(records, columns=cols)
        df = df.sort_values(["Obj1", "Obj2"])
        # Numerical duplicates close to the endpoints can be kept; they remain valid references.
        return df.reset_index(drop=True)

    def exact_box_sensitivity(self, x: np.ndarray,
                              delta_j: float | None = None) -> tuple[float, float]:
        """EXACT worst |Delta f1| and |Delta f2| for independent box perturbations.

        For f1 (convex quadratic): maximum over box is at one of 2^10 corners;
        minimum is the bounded least-squares solution. For f2, both extrema
        have simple closed forms. No model re-evaluation or MC needed.
        Bounds are clipped to the admissible design domain.
        """
        x = np.asarray(x, dtype=float).reshape(self.n_var)
        step = (self.s.independent_j_tolerance if delta_j is None else delta_j) / self.s.j_scale
        if step < 0:
            raise ValueError("delta_j must be nonnegative")
        lo = np.maximum(self.lower, x-step)
        hi = np.minimum(self.upper, x+step)
        if np.any(lo > hi):
            raise ValueError("x lies outside allowed design range")
        f1, f2 = self.evaluate(x)
        n = 1 << self.n_var
        vertices = ((np.arange(n, dtype=np.uint32)[:, None] >>
                     np.arange(self.n_var, dtype=np.uint32)) & 1)
        corners = lo + vertices*(hi-lo)
        residuals = self.K @ corners.T - self.b_target[:, None]
        f1_max = np.max(np.mean(residuals**2, axis=0))
        res_min = lsq_linear(self.K, self.b_target,
                             bounds=(lo-1e-14, hi+1e-14), tol=1e-11, max_iter=500)
        if not res_min.success:
            raise RuntimeError("Sensitivity minimum solve failed")
        f1_min = self.evaluate(res_min.x)[0]
        s1 = max(float(f1_max-f1), float(f1-f1_min), 0.0)
        farthest_abs = np.maximum(abs(lo), abs(hi))
        nearest_abs = np.where((lo <= 0) & (hi >= 0),
                               0.0, np.minimum(abs(lo), abs(hi)))
        f2_max = np.mean(farthest_abs**2)
        f2_min = np.mean(nearest_abs**2)
        s2 = max(float(f2_max-f2), float(f2-f2_min), 0.0)
        return s1, s2


if __name__ == "__main__":
    model = DiBarbaSolenoid()
    out = Path(__file__).resolve().parent / "results"
    out.mkdir(exist_ok=True)
    ref = model.reference_pareto(n_weights=101)
    path = out / "pareto_di_barba_analytical_reference.csv"
    ref.to_csv(path, index=False)
    print("Analytical solenoid model | independent normalized 10-D currents")
    print("A shape:", model.A.shape)
    print("Target field:", model.s.target_field * 1e3, "mT")
    print("Current density bounds:", model.s.j_min, model.s.j_max, "A/m^2")
    print("Power reference:", model.p_ref, "W")
    print("Zero-current objectives:", model.evaluate(np.zeros(model.n_var)))
    print("Reference solutions:", len(ref))
    print("Reference CSV:", path)
