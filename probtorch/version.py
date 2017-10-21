def git_revision():
    import subprocess
    rev = subprocess.check_output("git rev-parse --short HEAD".split())
    return rev.strip().decode('utf-8')

__version__ = "0.0+" + str(git_revision())
