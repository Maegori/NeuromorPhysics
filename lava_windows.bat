cd \Users\lexj1\Documents\RU\NMC\project\NeuroMorPhysics
.\.venv\Scripts\activate
git clone https://github.com/lava-nc/lava.git
cd lava
pip install poetry>=1.1.13
poetry config virtualenvs.in-project true
poetry install
pytest

cd ..
git clone https://github.com/lava-nc/lava-dl.git
cd lava-dl
poetry config virtualenvs.in-project true
poetry install
pytest