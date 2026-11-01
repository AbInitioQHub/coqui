"""
This test module checks pw2coqui.x on one and on several MPI ranks against reference files
from tests/unit_test_files/qe/.

It is a standalone test, not part of CoQui's Python test suite: it needs QE (pw2coqui.x on
PATH, or in QE_BIN_DIR when set) and CoQui (for coqui.TEST_INPUT_DIR), and must be run
with plain pytest, not under mpirun.
"""
import os
import shutil
import signal
import subprocess
from pathlib import Path

import numpy as np
import pytest

# pw2coqui.x from QE_BIN_DIR when set, otherwise from PATH (e.g. a loaded QE module)
QE_BIN_DIR = os.environ.get("QE_BIN_DIR", "")
PW2COQUI = Path(QE_BIN_DIR) / "pw2coqui.x" if QE_BIN_DIR else Path(shutil.which("pw2coqui.x") or "")
if not PW2COQUI.is_file():
  raise RuntimeError("pw2coqui.x not found: put it on PATH or set QE_BIN_DIR to its directory")

# Check the test is not executed under mpirun directly since it
# launches pw2coqui.x with mpirun itself under the hood.
if any(var in os.environ for var in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK")):
  raise RuntimeError("this test launches mpirun itself: run pytest without mpirun")

missing = [tool for tool in ("mpirun", "h5diff") if shutil.which(tool) is None]
if missing:
  raise RuntimeError(f"{' and '.join(missing)} not found on PATH")

import coqui
import h5

TEST_QE = Path(coqui.TEST_INPUT_DIR) / "qe"

# Test systems
SYSTEMS = [
  "lih_kp222_nbnd16_sym",  # norm-conserving, symmetry-reduced to 3 k-points
  "GaAs_kp222_so",         # spin-orbit (noncolin, lspinorb; npol = 2, nspin = 4), 8 k-points
]

# (id, MPI ranks, pw2coqui.x flags): the QE parallelization layouts under test
MPI_LAYOUTS = [
  ("np1", 1, []),                          # serial
  ("np2", 2, []),                          # G-space division
  ("np2_nk2", 2, ["-nk", "2"]),            # pools, one rank each
  ("np4_nk2", 4, ["-nk", "2"]),            # pools x G-space division
  ("np5_nk5", 5, ["-nk", "5"]),            # uneven pools; on 3 k-points two pools are empty
  ("np8_nk2_nb2", 8, ["-nk", "2", "-nb", "2"]),  # pools x band groups x G-space division
]

TIMEOUT = 120  # s; a collective called on a subset of ranks hangs forever


def _mpirun(nranks, qe_flags, input_file, cwd):
  """
  Run pw2coqui.x under mpirun and fail the test on a nonzero exit or a hang.
  """
  cmd = ["mpirun", "-np", str(nranks), str(PW2COQUI), *qe_flags, "-in", str(input_file)]
  env = dict(os.environ, OMP_NUM_THREADS="1")
  proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, start_new_session=True)
  try:
    stdout, stderr = proc.communicate(timeout=TIMEOUT)
  except subprocess.TimeoutExpired:
    os.killpg(proc.pid, signal.SIGTERM)
    try:
      stdout, _ = proc.communicate(timeout=30)
    except subprocess.TimeoutExpired:
      os.killpg(proc.pid, signal.SIGKILL)
      stdout, _ = proc.communicate()
    pytest.fail(f"{' '.join(cmd)} did not finish in {TIMEOUT} s (hang)\n{stdout[-3000:]}")
  assert proc.returncode == 0, (f"{' '.join(cmd)} exited {proc.returncode}\n"
                                f"{stdout[-3000:]}\n{stderr[-3000:]}")


CASES = [(system, *mpi_layout) for system in SYSTEMS for mpi_layout in MPI_LAYOUTS]


@pytest.fixture(params=CASES, ids=[f"{case[0]}-{case[1]}" for case in CASES])
def pw2coqui_run(request, tmp_path_factory):
  """
  Run pw2coqui.x for different systems under different MPI layouts.

  The run is copied into a temporary directory of its own, since pw2coqui.x writes into
  its outdir.
  """
  system, tag, nranks, qe_flags = request.param
  workdir = tmp_path_factory.mktemp(f"{system}-{tag}")
  shutil.copytree(TEST_QE / system / "pwscf.save", workdir / "out" / "pwscf.save")
  inp = workdir / "pw2coqui.in"
  inp.write_text(f"&input_pw2coqui\n  prefix = 'pwscf'\n  outdir = '{workdir / 'out'}'\n/\n")
  _mpirun(nranks, qe_flags, inp, workdir)
  output = workdir / "out" / "pwscf.coqui.h5"
  assert output.is_file(), f"pw2coqui.x exited 0 but wrote no {output}"
  return {"output": output, "reference": TEST_QE / system / "pwscf.coqui.h5"}


def _datasets(group, path=""):
  """
  Read every dataset below an h5 group, keyed by its path.
  """
  datasets = {}
  for key in group.keys():
    if group.is_group(key):
      datasets.update(_datasets(group[key], f"{path}/{key}"))
    else:
      datasets[f"{path}/{key}"] = group[key]
  return datasets


def test_pw2coqui(pw2coqui_run):
  """pw2coqui.x: the file written on any MPI layout matches the reference."""
  reference = _datasets(h5.HDFArchive(str(pw2coqui_run["reference"]), "r"))
  output = _datasets(h5.HDFArchive(str(pw2coqui_run["output"]), "r"))
  assert sorted(output) == sorted(reference)

  # tolerance for projectors is set to 1e-5 to accomodate the projector 
  # change in QE 7.6 (issue #802: https://gitlab.com/QEF/q-e/-/work_items/802). 
  tolerance, projector_tolerance = 1e-8, 1e-5

  # Datasets, with h5
  mismatches = []
  for path in sorted(reference):
    ref, out = np.asarray(reference[path]), np.asarray(output[path])
    if path.endswith("/atomic_id"):
      # The references store it 1-based, from before it became 0-based; CoQui's readers
      # accept both
      if not (np.array_equal(out, ref) or np.array_equal(out + 1, ref)):
        mismatches.append(f"{path}: {out} is neither {ref} nor {ref - 1}")
    elif ref.dtype.kind not in "fc" or ref.shape != out.shape:
      if not np.array_equal(out, ref):
        mismatches.append(f"{path}: differs")
    else:
      atol = projector_tolerance if "/projector_k" in path else tolerance
      max_diff = np.abs(out - ref).max(initial=0.0)
      if max_diff > atol:
        mismatches.append(f"{path}: max |diff| = {max_diff:.2e} > {atol:.0e}")
  assert not mismatches, "\n".join(mismatches)

  # Attributes, with h5diff: the h5 Python binding reads string attributes only. 
  # All dataset are excluded, which leaves the groups and their attributes.
  datasets = [arg for path in reference for arg in ("--exclude-path", path)]
  cmd = ["h5diff", "-r", "-d", str(tolerance), *datasets,
         str(pw2coqui_run["reference"]), str(pw2coqui_run["output"])]
  proc = subprocess.run(cmd, capture_output=True, text=True)
  assert proc.returncode == 0, (f"h5diff -r -d {tolerance} on the attributes (exit {proc.returncode}):\n"
                                f"{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
