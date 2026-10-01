
"""Python test configuration."""

# 3rd party libraries
import pytest
import logging

# own libraries
import transistordatabase as tdb

# Enable logger
logger = logging.getLogger(__name__)

@pytest.fixture(scope="session", autouse=True)
def database_updated():
    """Initialize the transistor database one time per test session."""
    # Initialize the transistor database and set the mode
    transistor_database = tdb.DatabaseManager()
    transistor_database.set_operation_mode_json()
    # Update the database from file exchange
    transistor_database.update_from_fileexchange()
    # Log database update information
    logger.info("Update transistor data from TDB file exchange.")
