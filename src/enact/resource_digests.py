"""Hashing functions for Enact resources."""
from typing import Any

import enact
from enact import digests

def resource_digest(value: Any) -> str:
  """Compute a hash digest of a resource."""
  return digests.digest(enact.wrap(value))
