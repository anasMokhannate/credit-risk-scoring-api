# Databricks notebook source
# MAGIC %md
# MAGIC # 00 - Download the Home Credit data from Kaggle into the Volume
# MAGIC Alternative to uploading the CSV files by hand: Databricks downloads them directly.
# MAGIC
# MAGIC **Before running**
# MAGIC 1. On kaggle.com, open *Settings > API* and create a token.
# MAGIC 2. Run the next two cells, then fill in the widgets at the top of the notebook:
# MAGIC    - a legacy token (`kaggle.json`): paste its `username` and `key`;
# MAGIC    - a new-style token (starts with `KGAT_`): leave the username empty, paste it as the key.
# MAGIC 3. Run the rest. Clear the widgets afterwards: the token must never be saved in the notebook.
# MAGIC
# MAGIC Your Kaggle account must already have accepted the competition rules (it has, if you
# MAGIC were able to download the data from the website).

# COMMAND ----------

# MAGIC %pip install -q kaggle
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("kaggle_username", "")  # noqa: F821 (dbutils is provided by Databricks)
dbutils.widgets.text("kaggle_key", "")  # noqa: F821

# COMMAND ----------

import os
import shutil
import sys
import tempfile
import zipfile

sys.path.append(os.path.abspath("../src"))

from credit_risk.config import SOURCE_TABLES, load_config

COMPETITION = "home-credit-default-risk"
config = load_config()

username = dbutils.widgets.get("kaggle_username").strip().strip("\"'")  # noqa: F821
key = dbutils.widgets.get("kaggle_key").strip().strip("\"'")  # noqa: F821
assert key, "Fill in the kaggle_key widget at the top of the notebook, then run this cell again."

# Start clean: credentials left over from an earlier attempt would be mixed with the new ones.
for variable in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
    os.environ.pop(variable, None)
new_style = key.startswith("KGAT_")
if new_style:
    os.environ["KAGGLE_API_TOKEN"] = key
else:
    assert username, "A legacy key needs the kaggle_username widget too (your Kaggle username)."
    assert "@" not in username, "Use your Kaggle username, not your email address."
    os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = username, key
print(f"Using a {'new-style token' if new_style else f'legacy key for user {username!r}'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the catalog, schema and volume if they do not exist

# COMMAND ----------

catalog = config.catalog
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")  # noqa: F821 (spark is provided by Databricks)
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.bronze")  # noqa: F821
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.bronze.raw")  # noqa: F821
print(f"Files will be written to {config.raw_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Download the archive and extract the eight CSV files into the Volume

# COMMAND ----------

from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Check the credentials on a small request before starting the 700 MB download.
try:
    api.competition_list_files(COMPETITION)
except Exception as error:
    status = getattr(getattr(error, "response", None), "status_code", None)
    reasons = {
        401: "Kaggle rejected the credentials. Create a new token on kaggle.com (Settings > API), "
        "paste it again without quotes or spaces, and rerun from the credentials cell.",
        403: "The credentials work but this account has not accepted the competition rules: "
        f"open https://www.kaggle.com/competitions/{COMPETITION}/rules and accept them.",
    }
    raise RuntimeError(reasons.get(status, f"Kaggle could not be reached: {error}")) from error
print("Kaggle credentials accepted")

workdir = tempfile.mkdtemp()
# One zip of about 700 MB; it stays on the compute's local disk and is deleted at the end.
api.competition_download_files(COMPETITION, path=workdir, quiet=False)
archive = os.path.join(workdir, f"{COMPETITION}.zip")
print(f"Downloaded {os.path.getsize(archive) / 1e6:,.0f} MB")

# COMMAND ----------

with zipfile.ZipFile(archive) as bundle:
    for name in SOURCE_TABLES:
        member = bundle.getinfo(f"{name}.csv")
        target = config.raw_file(name)
        if os.path.exists(target) and os.path.getsize(target) == member.file_size:
            print(f"already there  {target}")
            continue
        with bundle.open(member) as source, open(target, "wb") as destination:
            shutil.copyfileobj(source, destination, length=16 * 1024 * 1024)
        print(f"written        {target}  ({member.file_size / 1e6:,.0f} MB)")

shutil.rmtree(workdir)

# COMMAND ----------

display(dbutils.fs.ls(config.raw_path))  # noqa: F821

# COMMAND ----------

# Remove the token from the notebook's widgets.
dbutils.widgets.removeAll()  # noqa: F821
