"""
API layer for the MLOps sentiment analysis service.

This module contains all API-related components including routes,
middleware, and request/response schemas.

Note: deliberately no eager imports here. `app.api.routes` (transitively)
imports service modules that themselves import from `app.api.schemas`, a
submodule of this package. Importing `app.api.routes` here would force this
package's __init__ to run before those submodules finish loading, creating a
circular import whenever a service module is imported directly (e.g. in
tests). Import `app.api.routes` (or its `router`) directly instead.
"""
