import subprocess
import bittensor as bt


def get_current_branch():
    """Get the name of the current git branch"""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Failed to get current branch: {e}")
        return None


def is_up_to_date_with_main():
    """
    Check if the current branch is up to date with the latest commit on main.
    Returns True if up to date, False otherwise.
    """
    try:
        # Fetch the latest from origin to ensure we have the latest main
        bt.logging.info("Fetching latest commits from origin/main")
        subprocess.run(
            ["git", "fetch", "origin", "main", "--quiet"],
            stderr=subprocess.STDOUT,
            check=True,
        )

        # Get the commit hash of the latest commit on origin/main
        bt.logging.info("Getting latest commit hash from origin/main")
        main_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "origin/main"], stderr=subprocess.STDOUT
            )
            .decode("utf-8")
            .strip()
        )

        # Check if the current branch contains this commit
        bt.logging.info(f"Checking if current HEAD contains commit {main_commit[:8]}")
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", main_commit, "HEAD"],
            stderr=subprocess.STDOUT,
        )

        # Return True if the current branch contains the latest main commit
        is_up_to_date = result.returncode == 0
        if is_up_to_date:
            bt.logging.info(
                f"Current branch is up to date with origin/main ({main_commit[:8]})"
            )
        else:
            bt.logging.error(
                f"Current branch does NOT contain the latest commit from origin/main ({main_commit[:8]})"
            )
        return is_up_to_date
    except subprocess.CalledProcessError as e:
        bt.logging.error(f"Error checking if up to date with main: {e}")
        return False


def check_latest_code():
    """
    Checks if the repository is on the latest code from main branch.
    Raises RuntimeError if not up to date.
    """
    current_branch = get_current_branch()
    if current_branch is None:
        raise RuntimeError(
            "Failed to determine current git branch. Is this a git repository?"
        )

    bt.logging.info(f"Current git branch: {current_branch}")

    if not is_up_to_date_with_main():
        error_msg = f"Repository is not up to date with the latest commit on main! Current branch: {current_branch}"
        bt.logging.error(error_msg)
        raise RuntimeError(error_msg)

    bt.logging.info("Verified repository is up to date with the latest code from main")
    return True
