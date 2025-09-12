import json
import os
from coqui._lib.eri_module import CholCoulomb


def make_chol_coulomb(mf, params):
  # Create output directory if it does not exist
  path = os.path.abspath(os.path.expanduser(params.get("path", "./")))
  if not os.path.exists(path):
    try:
      os.makedirs(path)
    except Exception as e:
      raise RuntimeError(f"make_chol_coulomb: Failed to create directory '{path}': {e}") from e
  elif not os.path.isdir(path):
    raise RuntimeError(f"make_chol_coulomb: params['path'] exists but is not a directory: {path}")

  return CholCoulomb(mf, json.dumps(params))

