"""Hashing functions for Enact resources."""
from typing import Any

from enact import digests
from enact import resource_registry


def resource_digest(value: Any) -> str:
  """Compute a hash digest of a resource."""
  return digests.digest(resource_registry.wrap(value))
