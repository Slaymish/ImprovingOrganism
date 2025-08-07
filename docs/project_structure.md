# Project Structure

The project is organized into the following directories:

- `src/`: Contains the core source code for the application.
  - `main.py`: The main FastAPI application.
  - `config.py`: Configuration settings.
  - `critic_module.py`: The critic module for scoring responses.
  - `latent_workspace.py`: The latent workspace for reasoning.
  - `llm_wrapper.py`: A wrapper for the language model.
  - `memory_module.py`: The memory module for storing conversations.
  - `updater.py`: The updater module for fine-tuning the model.
- `tests/`: Contains the test suite.
- `scripts/`: Contains utility scripts.
  - `start.sh`: A script to start the application.
  - `demo.py`: A script to demonstrate the application.
- `data/`: Contains the application's data, such as the memory database.
- `adapters/`: Contains the LoRA adapters.
- `logs/`: Contains log files.
- `docs/`: Contains project documentation.
- `Dockerfile`: The Dockerfile for building the application image.
- `docker-compose.yml`: The Docker Compose file for running the application.
- `.gitignore`: A list of files and directories to ignore in version control.
- `requirements.txt`: A list of Python dependencies.
- `requirements-dev.txt`: A list of Python dependencies for development.
- `README.md`: This file.
