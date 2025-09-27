import logging

def setup_logging(logfile: str = "lane_emden.log", level: int = logging.INFO) -> None:
    """
    Set up basic logging configuration for the Lane-Emden PINN project.

    Args:
        logfile (str): Path to the log file.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler()
        ]
    )

# Example usage:
# setup_logging()