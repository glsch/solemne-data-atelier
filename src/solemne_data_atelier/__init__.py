import os
from pathlib import Path
from importlib import resources
from dotenv import dotenv_values
import importlib

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

__version__ = "0.0.1"
__package_name__ = "solemne_data_atelier"

def __getattr__(name):
    if name == "setup_logging":
        return getattr(importlib.import_module("solemne_data_atelier.logging"), "setup_logging")
    raise AttributeError(f"module {__name__} has no attribute {name}")

def get_package_root() -> Path:
    return Path(__file__).parent.resolve()


def get_config():
    config_path = Path(__file__).parent / "config.toml"

    if config_path.exists():
        with open(config_path, "rb") as f:
            return tomllib.load(f)

    try:
        return tomllib.loads(resources.read_text(f"{__package_name__}", "config.toml"))
    except Exception as e:
        raise RuntimeError(f"Could not load config.toml: {e}")


def get_resource_path(resource_name: str) -> Path:
    package_root = get_package_root()
    direct_path = package_root / _cfg["paths"][resource_name]

    direct_path.parent.mkdir(parents=True, exist_ok=True)

    if direct_path.exists():
        return direct_path

    try:
        resource_path = resources.files(__package_name__).joinpath(_cfg["paths"][resource_name])
        resource_path.parent.mkdir(parents=True, exist_ok=True)
        return resource_path
    except Exception as e:
        raise RuntimeError(f"Could not locate resource {resource_name}: {e}")


_cfg = get_config()

DATA_DIR = get_resource_path("data").resolve()
OUTPUT_DIR = get_resource_path("output").resolve()
MODEL_DIR = get_resource_path("models").resolve()

env_path = Path.cwd() / ".env"
if env_path.exists():
    settings = dotenv_values(env_path)
    for k, v in settings.items():
        os.environ[k] = v
