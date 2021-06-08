#!/bin/bash
export MLFLOW_TRACKING_URI=databricks://eastus2
mlflow run . -b databricks  --backend-config databricks_cluster_spec.json --experiment-id 4183507275202782
