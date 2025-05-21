# main.py (Modified to bypass license and default to PRO tier)

import sys
import os
import logging
# import requests # REMOVED: No longer needed for license verification
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication, QMessageBox # QInputDialog REMOVED

# --- Import configuration ---
import config  # Needed for APP_DIR and setting config.TIER

# --- Import the main window ---
# This import happens AFTER config.TIER is potentially set later
# from annotator_window import AnnotatorWindow # Moved import lower

# --- Backend License Verification URL (REMOVED) ---
# BACKEND_VERIFY_URL = "https://snowball-license-backend-frsu.vercel.app/api/verify-license"

# Setup logger for main module
logger_main = logging.getLogger(__name__)


def verify_license_with_backend():
    """
    Bypasses license verification and defaults to PRO tier.
    Sets config.TIER.

    Returns:
        bool: Always True.
    """
    logger_main.info("License verification bypassed by modified main.py. Defaulting to PRO tier.")
    config.TIER = "PRO"  # <<< SET TIER TO PRO
    return True  # Always return true


if __name__ == "__main__":
    # --- Setup Logging ---
    log_path = "app_debug.log"
    try:
        # Ensure APP_DIR exists before trying to use it for logging
        if config.APP_DIR:
            os.makedirs(config.APP_DIR, exist_ok=True)
            log_path_candidate = os.path.join(config.APP_DIR, "app_debug.log")
            # Check if we can write to the directory
            if os.access(config.APP_DIR, os.W_OK):
                log_path = log_path_candidate
            else:
                print(
                    f"[WARNING] No write access to {config.APP_DIR}. Logging to current directory."
                )
        else:
            print("[WARNING] config.APP_DIR not defined. Logging to current directory.")

        log_handlers = [logging.StreamHandler()]  # Always log to console
        log_handlers.append(
            logging.FileHandler(log_path, mode="a")
        )  # Append to log file

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s [%(levelname)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=log_handlers,
            force=True,  # Override any existing logging config
        )
        # Define logger_main AFTER basicConfig
        logger_main = logging.getLogger(__name__)
        logger_main.info(f"--- Application Started ---")  # Log start first
        logger_main.info(f"Version: {config.VERSION}")  # Then version
        logger_main.info(f"Logging initialized. Log file: {os.path.abspath(log_path)}")

    except Exception as log_ex:
        print(f"[CRITICAL] Failed to initialize logging: {log_ex}")
        # Ensure logger_main exists even if file logging fails
        if "logger_main" not in locals():
            logging.basicConfig(level=logging.INFO)  # Basic console logging
            logger_main = logging.getLogger(__name__)
            logger_main.error("File logging failed, using basic console logging.")

    # --- Run Application ---
    app = QApplication(sys.argv)

    logger_main.info(f"Bypassing license verification as per modified main.py.")
    # --- Call the updated function ---
    # It now returns only True, and sets config.TIER internally to "PRO"
    license_ok = verify_license_with_backend()

    if license_ok: # This will now always be true
        # --- config.TIER should now be "PRO" ---
        logger_main.info(
            f"Tier set to: {config.TIER}. Initializing main window..."
        )

        # --- Import AnnotatorWindow *after* config.TIER is set ---
        # This allows AnnotatorWindow's conditional imports to work correctly
        try:
            from annotator_window import AnnotatorWindow
        except ImportError as e:
            logger_main.critical(
                f"Failed to import AnnotatorWindow: {e}", exc_info=True
            )
            QMessageBox.critical(
                None, "Startup Error", f"Failed to import main window components:\n{e}"
            )
            sys.exit(1)
        except Exception as e_aw_import:
            logger_main.critical(
                f"Error importing AnnotatorWindow: {e_aw_import}", exc_info=True
            )
            QMessageBox.critical(
                None, "Startup Error", f"Error loading main window:\n{e_aw_import}"
            )
            sys.exit(1)

        # --- Initialize and Show Window ---
        try:
            # AnnotatorWindow __init__ will now read config.TIER
            window = AnnotatorWindow()
            window.show()
            logger_main.info("Main window displayed. Starting event loop.")
            sys.exit(app.exec())  # Start the Qt event loop
        except Exception as main_win_err:
            logger_main.exception("CRITICAL ERROR initializing or showing main window:")
            QMessageBox.critical(
                None,
                "Application Error",
                f"Failed to start the main application window:\n{main_win_err}",
            )
            sys.exit(1)
    else:
        # This 'else' branch is now effectively unreachable due to verify_license_with_backend() always returning True.
        # Kept for structural completeness, but behaviorally it won't be hit.
        logger_main.error(
            "License verification reported failure (should not happen with modified main.py). Exiting."
        )
        sys.exit(1)  # Exit the application cleanly