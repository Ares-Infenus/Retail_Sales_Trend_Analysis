from ..src.config import config, ConfigError

# Ejemplo de uso:
db_host = config["database"]["host"]
log_dir = config["paths"]["log_dir"]
log_level = config["other"].get("log_level", "WARNING")
