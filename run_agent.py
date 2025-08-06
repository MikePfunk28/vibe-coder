import subprocess
import os

query = "For Trusted by industry leaders, add two more boxes."
working_dir = ".\\project"

python_executable = os.path.join(".venv", "Scripts", "python.exe")
main_script = "main.py"

command = [python_executable, main_script, "--query", query, "--working-dir", working_dir]

print(f"Running command: {' '.join(command)}")

result = subprocess.run(command, capture_output=True, text=True, cwd=os.getcwd())

print("Stdout:")
print(result.stdout)
print("Stderr:")
print(result.stderr)
print(f"Exit Code: {result.returncode}")