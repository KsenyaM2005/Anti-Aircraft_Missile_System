import logging
import os

# Ensure logs directory exists
os.makedirs("./logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    filename="./logs/simulation.log",
    filemode="w",
    format="%(asctime)s %(levelname)s [%(name)s]: %(message)s"
)

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


# Pre-configured loggers for each module
environment_logger = get_logger("Environment")
radar_logger = get_logger("Radar")
pbu_logger = get_logger("PBU")
launcher_logger = get_logger("Launcher")
missile_logger = get_logger("Missile")
dispatcher_logger = get_logger("Dispatcher")
gui_logger = get_logger("GUI")