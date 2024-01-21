import subprocess
from flask import Flask, jsonify
import sys
import os

app = Flask(__name__)


@app.route('/api/run-script', methods=['GET'])
def run_script():
    try:
        # Get the path to the virtual environment (venv)
        venv_path = os.path.join('venv\Scripts')

        # Construct the activate script path based on the operating system
        activate_script = 'activate' if sys.platform == 'win32' else 'bin/activate'

        # Activate the virtual environment
        activate_venv_command = os.path.join(venv_path, activate_script)

        # Construct the full command to run the script
        script_command = f'{activate_venv_command} && python main.py'

        # Run the script
        subprocess.run(script_command, check=True, shell=True)

        message = 'Script executed successfully'
        status_code = 200
    except subprocess.CalledProcessError:
        message = 'Error executing the script'
        status_code = 500

    return jsonify({'message': message}), status_code


if __name__ == '__main__':
    app.run(debug=True, port=5001)

# http://127.0.0.1:5001/api/run-script