import subprocess
import sys
import pkg_resources

def install_missing_packages(requirements_file="requirements.txt"):
    # Get installed packages
    installed_packages = {pkg.key for pkg in pkg_resources.working_set}

    # Read required packages
    with open(requirements_file) as f:
        required_packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # Find missing packages
    missing_packages = []
    for package in required_packages:
        pkg_name = package.split("==")[0].lower()
        if pkg_name not in installed_packages:
            missing_packages.append(package)

    # Install missing ones
    if missing_packages:
        print("Installing missing packages:", missing_packages)
        subprocess.check_call([sys.executable, "-m", "pip", "install", *missing_packages])
    else:
        print("All packages already installed ✅")

if __name__ == "__main__":
    install_missing_packages()