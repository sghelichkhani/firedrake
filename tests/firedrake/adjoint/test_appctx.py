"""Tests that appctx is correctly cloned for forward and adjoint
solver replays during tape evaluation.

When a NonlinearVariationalSolver is created with an appctx dict,
the adjoint machinery creates clone solvers for tape replay. The
clone solvers' form coefficients (F, J) are deep-copied so they can
be independently updated from the tape. The appctx dict must undergo
the same replacement so that preconditioners reading from appctx
(e.g. MassInvPC) see the cloned values that _ad_solver_replace_forms
keeps in sync with the tape.
"""
import pytest
import numpy as np

from firedrake import *
from firedrake.adjoint import *
from firedrake.adjoint_utils.blocks import NonlinearVariationalSolveBlock
from firedrake.preconditioners.base import PCBase
from firedrake.dmhooks import get_function_space


@pytest.fixture(autouse=True)
def autouse_set_test_tape(set_test_tape):
    pass


class MuRecorderPC(PCBase):
    """Minimal PC that records the appctx 'mu' value at every
    initialize/update. Used to verify appctx liveness during replay."""
    _prefix = "recorder_"
    mu_log = []

    def initialize(self, pc):
        from firedrake.assemble import get_assembler
        appctx = self.get_appctx(pc)
        self._mu_ref = appctx["mu"]

        V = get_function_space(pc.getDM())
        test, trial = TestFunction(V), TrialFunction(V)
        a = inner(trial, test) * dx
        assembler = get_assembler(a, mat_type="aij")
        self._P = assembler.allocate()
        self._assemble = assembler.assemble
        self._assemble(tensor=self._P)

        from firedrake.petsc import PETSc
        inner_pc = PETSc.PC().create(comm=pc.comm)
        inner_pc.setOperators(self._P.petscmat, self._P.petscmat)
        inner_pc.setType("jacobi")
        inner_pc.setUp()
        self._pc = inner_pc
        self._record()

    def update(self, pc):
        self._assemble(tensor=self._P)
        self._pc.setUp()
        self._record()

    def _record(self):
        mu = self._mu_ref
        if isinstance(mu, Function):
            val = float(mu.dat.data_ro[0])
        elif isinstance(mu, Constant):
            val = float(mu)
        else:
            val = None
        MuRecorderPC.mu_log.append(val)

    def apply(self, pc, x, y):
        self._pc.apply(x, y)

    def applyTranspose(self, pc, x, y):
        self._pc.applyTranspose(x, y)


def _setup_problem_with_appctx(solver_params=None):
    """Build a Poisson-like problem where mu appears in both the form
    and the appctx.  Two solves are performed with mu=1 and mu=10 so
    that stale-appctx bugs are detectable."""
    mesh = UnitSquareMesh(4, 4)
    V = FunctionSpace(mesh, "CG", 1)

    mu = Function(V, name="mu")
    mu.assign(1.0)

    u = Function(V, name="u")
    v = TestFunction(V)

    F = mu * inner(grad(u), grad(v)) * dx - Constant(1.0) * v * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    if solver_params is None:
        solver_params = {"snes_type": "ksponly",
                         "ksp_type": "cg", "pc_type": "sor"}

    problem = NonlinearVariationalProblem(F, u, bcs=bc)
    solver = NonlinearVariationalSolver(
        problem, solver_parameters=solver_params, appctx={"mu": mu})

    control = Function(V, name="control")
    control.assign(1.0)

    mu.assign(control)
    solver.solve()
    mu.assign(10.0 * control)
    solver.solve()

    J = assemble(u * u * dx)
    return J, control, mu


@pytest.mark.skipcomplex
def test_appctx_forward_clone_identity():
    """The forward clone solver's appctx['mu'] must be the same object
    as the mu coefficient inside the clone's F form, so that
    _ad_solver_replace_forms keeps them in sync during replay."""
    J, control, mu = _setup_problem_with_appctx()

    tape = get_working_tape()
    solve_blocks = [b for b in tape.get_blocks()
                    if isinstance(b, NonlinearVariationalSolveBlock)]
    assert len(solve_blocks) >= 2

    clone = solve_blocks[0]._ad_solvers["forward_nlvs"]
    assert clone is not None

    appctx_mu = clone._ad_kwargs["appctx"]["mu"]
    assert appctx_mu is not mu

    clone_F_coeffs = clone._problem.F.coefficients()
    clone_mu_candidates = [c for c in clone_F_coeffs if c.name() == "mu"]
    assert len(clone_mu_candidates) == 1
    clone_mu = clone_mu_candidates[0]
    assert clone_mu is not mu
    assert appctx_mu is clone_mu


@pytest.mark.skipcomplex
def test_appctx_adjoint_clone_identity():
    """The adjoint clone solver's appctx['mu'] must be the same object
    as the mu coefficient inside the adjoint's J form (= adjoint of
    dF/du). It must be distinct from both the original and the forward
    clone, since _ad_solver_replace_forms(Solver.ADJOINT) updates it
    independently."""
    J, control, mu = _setup_problem_with_appctx()

    Jhat = ReducedFunctional(J, Control(control))
    dJ = Jhat.derivative()

    tape = get_working_tape()
    solve_blocks = [b for b in tape.get_blocks()
                    if isinstance(b, NonlinearVariationalSolveBlock)]
    assert len(solve_blocks) >= 2

    adj_lvs = solve_blocks[0]._ad_solvers["adjoint_lvs"]
    assert adj_lvs is not None

    fwd_clone = solve_blocks[0]._ad_solvers["forward_nlvs"]
    fwd_appctx_mu = fwd_clone._ad_kwargs["appctx"]["mu"]

    adj_appctx = adj_lvs._ctx.appctx
    assert "mu" in adj_appctx
    adj_appctx_mu = adj_appctx["mu"]

    assert adj_appctx_mu is not mu
    assert adj_appctx_mu is not fwd_appctx_mu

    adj_J_coeffs = adj_lvs._problem.J.coefficients()
    adj_mu_candidates = [c for c in adj_J_coeffs if c.name() == "mu"]
    assert len(adj_mu_candidates) == 1
    assert adj_appctx_mu is adj_mu_candidates[0]


@pytest.mark.skipcomplex
def test_appctx_forward_replay():
    """During forward tape replay the preconditioner must see the same
    mu values as the original forward run, not the stale end-of-run
    value."""
    MuRecorderPC.mu_log.clear()

    solver_params = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": __name__ + ".MuRecorderPC",
    }
    J, control, mu = _setup_problem_with_appctx(solver_params)
    fwd_log = list(MuRecorderPC.mu_log)

    MuRecorderPC.mu_log.clear()
    Jhat = ReducedFunctional(J, Control(control))
    Jhat(control)

    replay_log = list(MuRecorderPC.mu_log)
    assert fwd_log == replay_log


@pytest.mark.skipcomplex
def test_appctx_adjoint_replay():
    """During the adjoint pass the preconditioner must see the correct
    per-step mu from the forward trajectory, not the stale end-of-run
    value. The adjoint traverses the tape backward, so both mu=1 and
    mu=10 must appear in the log."""
    MuRecorderPC.mu_log.clear()

    solver_params = {
        "snes_type": "ksponly",
        "ksp_type": "cg",
        "pc_type": "python",
        "pc_python_type": __name__ + ".MuRecorderPC",
    }
    J, control, mu = _setup_problem_with_appctx(solver_params)
    fwd_log = list(MuRecorderPC.mu_log)

    MuRecorderPC.mu_log.clear()
    Jhat = ReducedFunctional(J, Control(control))
    dJ = Jhat.derivative()

    adj_log = list(MuRecorderPC.mu_log)
    adj_values = set(round(v, 1) for v in adj_log if v is not None)
    assert 1.0 in adj_values, (
        f"adjoint PC never saw mu=1.0 (log: {adj_log})")
    assert 10.0 in adj_values, (
        f"adjoint PC never saw mu=10.0 (log: {adj_log})")


@pytest.mark.skipcomplex
def test_appctx_taylor_test():
    """Taylor test with a mu-weighted Poisson problem and appctx.
    Verifies that the adjoint gradient is correct when appctx is used,
    which requires appctx cloning to be correct for both the forward
    replay and the adjoint solve."""
    J, control, mu = _setup_problem_with_appctx()

    Jhat = ReducedFunctional(J, Control(control))
    h = Function(control.function_space())
    h.assign(1.0)
    assert taylor_test(Jhat, control, h) > 1.9
