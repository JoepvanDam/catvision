# Data labellen
Run the following:

```bash
pip install label-studio
label-studio start
```

* Open Label Studio at http://localhost:8080
* Sign up (or sign in)
* Create project
* Name the project
* Click data import, select all images
* Click labeling setup, enter your label(s)
* Save project and start labeling!

# Gebruik van catvision test/train.py met venv
```bash
wsl~
sudo apt install python3-venv
python3 -m venv catvision
source catvision/bin/activate
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip3 install matplotlib pillow tqdm flask
sudo apt-get install python3-tk  # TKinter install
cd /mnt/PATH_TO_CURRENT_FOLDER # Kan ook zonder
python3 PATH_TO_FILE # Relatief vanaf huidige folder
```

# Gebruik van catvision test/train.py zonder venv
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install matplotlib pillow tqdm flask
python PATH_TO_FILE # Relatief vanaf huidige folder
```