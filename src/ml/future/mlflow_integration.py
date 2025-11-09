"""
MLflow Integration for Advanced ML Model Management

This module provides comprehensive MLflow integration for:
- Model versioning and tracking
- Experiment management
- Model registry
- Automated model deployment
"""

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.tensorflow
import mlflow.xgboost
import mlflow.lightgbm
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import pickle
import shutil
from pathlib import Path
from dataclasses import  asdict

from src.model.machine_learning import ModelMetadata
from src.notification.logger import setup_logger

logger = setup_logger(__name__)


class MLflowManager:
    """Manages MLflow integration for model lifecycle."""

    def __init__(self,
                 tracking_uri: str = "sqlite:///mlflow.db",
                 registry_uri: str = "sqlite:///mlflow.db",
                 artifacts_dir: str = "mlruns",
                 experiment_name: str = "crypto_trading"):

        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.artifacts_dir = artifacts_dir
        self.experiment_name = experiment_name

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(registry_uri)

        # Create experiment if it doesn't exist
        self._setup_experiment()

        # Initialize model registry client
        self.client = mlflow.tracking.MlflowClient()

        logger.info("MLflow initialized with tracking URI: %s", tracking_uri)

    def _setup_experiment(self):
        """Setup MLflow experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                mlflow.create_experiment(self.experiment_name)
                logger.info("Created experiment: %s", self.experiment_name)
            else:
                mlflow.set_experiment(self.experiment_name)
                logger.info("Using existing experiment: %s", self.experiment_name)
        except Exception:
            logger.exception("Error setting up experiment: ")
            raise

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None) -> str:
        """Start a new MLflow run."""
        try:
            mlflow.start_run(run_name=run_name)
            run_id = mlflow.active_run().info.run_id

            # Add tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)

            # Add system tags
            mlflow.set_tag("framework", "crypto_trading")
            mlflow.set_tag("created_at", datetime.now().isoformat())

            logger.info("Started MLflow run: %s", run_id)
            return run_id

        except Exception:
            logger.exception("Error starting MLflow run: ")
            raise

    def end_run(self):
        """End the current MLflow run."""
        try:
            mlflow.end_run()
            logger.info("Ended MLflow run")
        except Exception:
            logger.exception("Error ending MLflow run: ")
            raise

    def log_parameters(self, params: Dict[str, Any]):
        """Log hyperparameters to MLflow."""
        try:
            mlflow.log_params(params)
            logger.info("Logged %d parameters", len(params))
        except Exception:
            logger.exception("Error logging parameters: ")
            raise

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics to MLflow."""
        try:
            mlflow.log_metrics(metrics, step=step)
            logger.info("Logged %d metrics", len(metrics))
        except Exception:
            logger.exception("Error logging metrics: ")
            raise

    def log_model(self,
                  model,
                  model_name: str,
                  model_type: str,
                  metadata: ModelMetadata,
                  artifacts: Dict[str, str] = None):
        """Log model to MLflow with metadata."""
        try:
            # Log model based on type
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, model_name, registered_model_name=model_name)
            elif model_type == "tensorflow":
                mlflow.tensorflow.log_model(model, model_name, registered_model_name=model_name)
            elif model_type == "xgboost":
                mlflow.xgboost.log_model(model, model_name, registered_model_name=model_name)
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model, model_name, registered_model_name=model_name)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            # Log metadata
            mlflow.log_dict(asdict(metadata), f"{model_name}_metadata.json")

            # Log artifacts
            if artifacts:
                for name, path in artifacts.items():
                    mlflow.log_artifact(path, name)

            logger.info("Logged model: %s (type: %s)", model_name, model_type)

        except Exception:
            logger.exception("Error logging model: ")
            raise

    def register_model(self,
                      model_name: str,
                      model_uri: str,
                      stage: str = "Staging",
                      description: str = None):
        """Register model in MLflow Model Registry."""
        try:
            # Create model if it doesn't exist
            try:
                self.client.create_registered_model(model_name, description)
            except:
                pass  # Model already exists

            # Create new model version
            model_version = self.client.create_model_version(
                name=model_name,
                source=model_uri,
                run_id=mlflow.active_run().info.run_id
            )

            # Transition to specified stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )

            logger.info("Registered model: %s v%s -> %s", model_name, model_version.version, stage)
            return model_version.version

        except Exception:
            logger.exception("Error registering model: ")
            raise

    def load_model(self, model_name: str, stage: str = "Production") -> Any:
        """Load model from MLflow Model Registry."""
        try:
            model_uri = f"models:/{model_name}/{stage}"

            # Try to load based on model type (you might need to store this info)
            try:
                model = mlflow.sklearn.load_model(model_uri)
                logger.info("Loaded sklearn model: %s", model_name)
                return model
            except:
                pass

            try:
                model = mlflow.pytorch.load_model(model_uri)
                logger.info("Loaded PyTorch model: %s", model_name)
                return model
            except:
                pass

            try:
                model = mlflow.tensorflow.load_model(model_uri)
                logger.info("Loaded TensorFlow model: %s", model_name)
                return model
            except:
                pass

            try:
                model = mlflow.xgboost.load_model(model_uri)
                logger.info("Loaded XGBoost model: %s", model_name)
                return model
            except:
                pass

            try:
                model = mlflow.lightgbm.load_model(model_uri)
                logger.info("Loaded LightGBM model: %s", model_name)
                return model
            except:
                pass

            raise ValueError(f"Could not load model: {model_name}")

        except Exception:
            logger.exception("Error loading model: ")
            raise

    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models."""
        try:
            models = self.client.list_registered_models()
            return [
                {
                    "name": model.name,
                    "creation_timestamp": model.creation_timestamp,
                    "last_updated_timestamp": model.last_updated_timestamp,
                    "description": model.description,
                    "latest_versions": [
                        {
                            "version": v.version,
                            "stage": v.current_stage,
                            "run_id": v.run_id,
                            "status": v.status
                        }
                        for v in model.latest_versions
                    ]
                }
                for model in models
            ]
        except Exception:
            logger.exception("Error listing models: ")
            return []

    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a specific model."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            return [
                {
                    "version": v.version,
                    "stage": v.current_stage,
                    "run_id": v.run_id,
                    "status": v.status,
                    "creation_timestamp": v.creation_timestamp
                }
                for v in versions
            ]
        except Exception:
            logger.exception("Error getting model versions: ")
            return []

    def promote_model(self, model_name: str, version: int, stage: str):
        """Promote model to a specific stage."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info("Promoted model %s v%s to %s", model_name, version, stage)
        except Exception:
            logger.exception("Error promoting model: ")
            raise

    def archive_model(self, model_name: str, version: int):
        """Archive a model version."""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Archived"
            )
            logger.info("Archived model %s v%s", model_name, version)
        except Exception:
            logger.exception("Error archiving model: ")
            raise

    def delete_model(self, model_name: str):
        """Delete a model from registry."""
        try:
            self.client.delete_registered_model(model_name)
            logger.info("Deleted model: %s", model_name)
        except Exception:
            logger.exception("Error deleting model: ")
            raise


class ModelDeployer:
    """Handles automated model deployment."""

    def __init__(self, deployment_config: Dict[str, Any]):
        self.config = deployment_config
        self.deployment_dir = deployment_config.get("deployment_dir", "deployments")
        self.backup_dir = deployment_config.get("backup_dir", "backups")

        # Create directories
        Path(self.deployment_dir).mkdir(exist_ok=True)
        Path(self.backup_dir).mkdir(exist_ok=True)

    def deploy_model(self,
                    model_name: str,
                    model_version: int,
                    mlflow_manager: MLflowManager,
                    deployment_type: str = "rolling") -> bool:
        """Deploy model to production."""
        try:
            if deployment_type == "rolling":
                return self._rolling_deployment(model_name, model_version, mlflow_manager)
            elif deployment_type == "blue_green":
                return self._blue_green_deployment(model_name, model_version, mlflow_manager)
            else:
                raise ValueError(f"Unsupported deployment type: {deployment_type}")

        except Exception:
            logger.exception("Error deploying model: ")
            return False

    def _rolling_deployment(self,
                          model_name: str,
                          model_version: int,
                          mlflow_manager: MLflowManager) -> bool:
        """Perform rolling deployment with zero downtime."""
        try:
            # Load new model
            model = mlflow_manager.load_model(model_name, "Staging")

            # Create deployment package
            deployment_package = self._create_deployment_package(
                model_name, model_version, model
            )

            # Backup current deployment
            self._backup_current_deployment(model_name)

            # Deploy new model
            deployment_path = f"{self.deployment_dir}/{model_name}"
            shutil.rmtree(deployment_path, ignore_errors=True)
            shutil.copytree(deployment_package, deployment_path)

            # Update deployment metadata
            self._update_deployment_metadata(model_name, model_version)

            # Promote model to production
            mlflow_manager.promote_model(model_name, model_version, "Production")

            logger.info("Successfully deployed %s v%s", model_name, model_version)
            return True

        except Exception:
            logger.exception("Rolling deployment failed: ")
            # Rollback
            self._rollback_deployment(model_name)
            return False

    def _blue_green_deployment(self,
                             model_name: str,
                             model_version: int,
                             mlflow_manager: MLflowManager) -> bool:
        """Perform blue-green deployment."""
        try:
            # Determine current environment (blue or green)
            current_env = self._get_current_environment(model_name)
            new_env = "green" if current_env == "blue" else "blue"

            # Deploy to new environment
            model = mlflow_manager.load_model(model_name, "Staging")
            deployment_package = self._create_deployment_package(
                model_name, model_version, model
            )

            new_deployment_path = f"{self.deployment_dir}/{model_name}_{new_env}"
            shutil.rmtree(new_deployment_path, ignore_errors=True)
            shutil.copytree(deployment_package, new_deployment_path)

            # Test new deployment
            if not self._test_deployment(new_deployment_path):
                logger.error("New deployment test failed")
                return False

            # Switch traffic to new environment
            self._switch_traffic(model_name, new_env)

            # Promote model to production
            mlflow_manager.promote_model(model_name, model_version, "Production")

            logger.info("Successfully deployed %s v%s to %s", model_name, model_version, new_env)
            return True

        except Exception:
            logger.exception("Blue-green deployment failed: ")
            return False

    def _create_deployment_package(self,
                                 model_name: str,
                                 model_version: int,
                                 model: Any) -> str:
        """Create deployment package with model and dependencies."""
        package_dir = f"temp_deployment_{model_name}_{model_version}"
        Path(package_dir).mkdir(exist_ok=True)

        # Save model
        model_path = f"{package_dir}/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Create deployment config
        deployment_config = {
            "model_name": model_name,
            "model_version": model_version,
            "deployed_at": datetime.now().isoformat(),
            "model_path": "model.pkl"
        }

        with open(f"{package_dir}/deployment_config.json", 'w') as f:
            json.dump(deployment_config, f, indent=2)

        return package_dir

    def _backup_current_deployment(self, model_name: str):
        """Backup current deployment."""
        current_path = f"{self.deployment_dir}/{model_name}"
        if os.path.exists(current_path):
            backup_path = f"{self.backup_dir}/{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(current_path, backup_path)

    def _rollback_deployment(self, model_name: str):
        """Rollback to previous deployment."""
        # Find latest backup
        backup_pattern = f"{self.backup_dir}/{model_name}_*"
        backups = sorted(Path(self.backup_dir).glob(f"{model_name}_*"))

        if backups:
            latest_backup = backups[-1]
            deployment_path = f"{self.deployment_dir}/{model_name}"
            shutil.rmtree(deployment_path, ignore_errors=True)
            shutil.copytree(latest_backup, deployment_path)
            logger.info("Rolled back %s to %s", model_name, latest_backup)

    def _test_deployment(self, deployment_path: str) -> bool:
        """Test deployment with sample data."""
        try:
            # Load model
            with open(f"{deployment_path}/model.pkl", 'rb') as f:
                model = pickle.load(f)

            # Create sample data for testing
            sample_data = np.random.randn(10, 5)  # Adjust based on your model

            # Test prediction
            if hasattr(model, 'predict'):
                prediction = model.predict(sample_data)
                return True
            else:
                return False

        except Exception:
            logger.exception("Deployment test failed: ")
            return False

    def _get_current_environment(self, model_name: str) -> str:
        """Get current deployment environment."""
        # Check which environment is currently active
        blue_path = f"{self.deployment_dir}/{model_name}_blue"
        green_path = f"{self.deployment_dir}/{model_name}_green"

        if os.path.exists(blue_path):
            return "blue"
        elif os.path.exists(green_path):
            return "green"
        else:
            return "blue"  # Default

    def _switch_traffic(self, model_name: str, new_env: str):
        """Switch traffic to new environment."""
        # Update load balancer or routing configuration
        # This is a simplified implementation
        current_env = "green" if new_env == "blue" else "blue"

        # Update routing config
        routing_config = {
            "model_name": model_name,
            "active_environment": new_env,
            "switched_at": datetime.now().isoformat()
        }

        with open(f"{self.deployment_dir}/routing_config.json", 'w') as f:
            json.dump(routing_config, f, indent=2)

    def _update_deployment_metadata(self, model_name: str, model_version: int):
        """Update deployment metadata."""
        metadata = {
            "model_name": model_name,
            "model_version": model_version,
            "deployed_at": datetime.now().isoformat(),
            "deployment_type": "rolling"
        }

        with open(f"{self.deployment_dir}/{model_name}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)


class ExperimentManager:
    """Manages ML experiments and comparisons."""

    def __init__(self, mlflow_manager: MLflowManager):
        self.mlflow_manager = mlflow_manager

    def create_experiment(self,
                         experiment_name: str,
                         description: str = None,
                         tags: Dict[str, str] = None) -> str:
        """Create a new experiment."""
        try:
            experiment_id = mlflow.create_experiment(
                experiment_name,
                tags=tags
            )

            if description:
                self.mlflow_manager.client.set_experiment_tag(
                    experiment_id, "description", description
                )

            logger.info("Created experiment: %s (ID: %s)", experiment_name, experiment_id)
            return experiment_id

        except Exception:
            logger.exception("Error creating experiment: ")
            raise

    def compare_runs(self,
                    experiment_name: str,
                    metric_name: str = "test_accuracy",
                    max_results: int = 10) -> pd.DataFrame:
        """Compare runs within an experiment."""
        try:
            # Get experiment
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                raise ValueError(f"Experiment not found: {experiment_name}")

            # Search runs
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric_name} DESC"],
                max_results=max_results
            )

            # Extract relevant information
            comparison_data = []
            for _, run in runs.iterrows():
                comparison_data.append({
                    "run_id": run["run_id"],
                    "run_name": run["tags.mlflow.runName"],
                    "status": run["status"],
                    "start_time": run["start_time"],
                    "end_time": run["end_time"],
                    "metric": run[f"metrics.{metric_name}"],
                    "parameters": run.filter(like="params.").to_dict()
                })

            return pd.DataFrame(comparison_data)

        except Exception:
            logger.exception("Error comparing runs: ")
            return pd.DataFrame()

    def get_best_run(self,
                    experiment_name: str,
                    metric_name: str = "test_accuracy") -> Dict[str, Any]:
        """Get the best run from an experiment."""
        try:
            comparison_df = self.compare_runs(experiment_name, metric_name, max_results=1)

            if comparison_df.empty:
                return {}

            best_run = comparison_df.iloc[0]
            return {
                "run_id": best_run["run_id"],
                "run_name": best_run["run_name"],
                "metric_value": best_run["metric"],
                "parameters": best_run["parameters"]
            }

        except Exception:
            logger.exception("Error getting best run: ")
            return {}

    def export_experiment(self,
                         experiment_name: str,
                         export_path: str) -> bool:
        """Export experiment data to file."""
        try:
            comparison_df = self.compare_runs(experiment_name)

            if not comparison_df.empty:
                comparison_df.to_csv(f"{export_path}/{experiment_name}_runs.csv", index=False)
                logger.info("Exported experiment to %s", export_path)
                return True
            else:
                logger.warning("No runs found for experiment: %s", experiment_name)
                return False

        except Exception:
            logger.exception("Error exporting experiment: ")
            return False
