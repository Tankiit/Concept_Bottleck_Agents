"""
credal_bq
---------
Package for Credal Quadrature components used by train_credal.py.

This exposes kernels and models so callers can import:
  from credal_bq.kernels.cosine_kernels import ...
  from credal_bq.models.credal_bq import CredalBQ
  from credal_bq.models.bq_integrator import BQIntegrator
"""

from .models.credal_bq import CredalBQ, CQOutput  # re-export
from .models.bq_integrator import BQIntegrator, BQIntegratorWithNoise, _ensure_unit  # re-export

