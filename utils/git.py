from pathlib import Path
import subprocess


def git_describe(path: Path = Path(__file__).parent) -> str:
    """return human-readable git description
    i.e. v5.0-5-g3e25f1e https://git-scm.com/docs/git-describe

    Args:
        path (Path, optional): path must be a directory. Defaults to Path(__file__).parent (parent of current directory).

    Returns:
        str: git details of given paths. "" if given path is not a git repository
    """
    s = f"git -C {path} describe --tags --long --always"
    try:
        git_commit = subprocess.check_output(
            s, shell=True, stderr=subprocess.STDOUT
        ).decode()[:-1]
        return "git commit: " + git_commit
    except subprocess.CalledProcessError:
        return ""  # not a git repository
