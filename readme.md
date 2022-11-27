# NeuromorPhysics
Project containing code for running an RBM solving the quantum many body problem on loihi hardware.

## Installation
First, create a virtual environment and activate it:
```
python3 -m venv .venv
source .venv/bin/activate
```
Then, install the required packages:
```
pip install -r requirements.txt
```
Finally, install the packages for lava:
```
git clone https://github.com/lava-nc/lava.git
cd lava
pip install poetry
poetry config virtualenvs.in-project true
poetry install
pytest

cd ..
git clone https://github.com/lava-nc/lava-dl.git
cd lava-dl
poetry config virtualenvs.in-project true
poetry install
pytest
```