"""Pipeline configuration: where raw files live and how tables are named.

The same code runs in two environments:

* Databricks: tables are ``<catalog>.<schema>.<table>`` in Unity Catalog and raw
  CSVs sit in a Volume.
* Local: there is no catalog level, so tables are ``<schema>.<table>`` in a local
  Delta warehouse and raw CSVs sit in ``data/raw``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

LAYERS = ("bronze", "silver", "gold")

# The seven Home Credit source files (without the .csv extension).
SOURCE_TABLES = (
    "application_train",
    "application_test",
    "bureau",
    "bureau_balance",
    "previous_application",
    "POS_CASH_balance",
    "installments_payments",
    "credit_card_balance",
)


def on_databricks() -> bool:
    return "DATABRICKS_RUNTIME_VERSION" in os.environ


@dataclass(frozen=True)
class Config:
    raw_path: str
    catalog: str | None = None  # None locally (no three-level namespace)
    schema_prefix: str = ""  # lets tests and dev runs use isolated schemas

    def schema(self, layer: str) -> str:
        if layer not in LAYERS:
            raise ValueError(f"Unknown layer {layer!r}, expected one of {LAYERS}")
        name = f"{self.schema_prefix}{layer}"
        return f"{self.catalog}.{name}" if self.catalog else name

    def table(self, layer: str, name: str) -> str:
        return f"{self.schema(layer)}.{name.lower()}"

    @property
    def model_name(self) -> str:
        """Registered model name: three levels in Unity Catalog, a plain name locally."""
        return f"{self.catalog}.gold.pd_model" if self.catalog else "pd_model"

    def raw_file(self, name: str) -> str:
        return f"{self.raw_path.rstrip('/')}/{name}.csv"


def load_config() -> Config:
    """Build the config for the current environment, overridable by env vars."""
    if on_databricks():
        catalog = os.environ.get("CREDIT_RISK_CATALOG", "credit_risk")
        default_raw = f"/Volumes/{catalog}/bronze/raw"
    else:
        catalog = None
        default_raw = "data/raw"
    return Config(
        raw_path=os.environ.get("CREDIT_RISK_RAW_PATH", default_raw),
        catalog=catalog,
        schema_prefix=os.environ.get("CREDIT_RISK_SCHEMA_PREFIX", ""),
    )
