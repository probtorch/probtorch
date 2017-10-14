def git_revision():
    import subprocess
    return subprocess.check_output(
                "git rev-parse --short HEAD".split()).strip().decode('utf-8')

__version__ = "0.0+" + str(git_revision())