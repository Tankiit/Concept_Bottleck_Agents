from .bq_integrator import (
    BQIntegrator,
    BQIntegratorWithNoise,
    _ensure_unit,
)
from .credal_bq import CredalBQ, CQOutput
try:
    from .credal_cbm import ConceptEncoder as CBMConceptEncoder, CredalCBM
except Exception:
    # credal_cbm is optional for train_credal.py path
    CBMConceptEncoder = None
    CredalCBM = None

