# machine-learning-unict-20202021
[info](https://techbrij.com/setup-tensorflow-jupyter-notebook-vscode-deep-learning) set up

Aprendo la cartella si aprirà automaticamente il venv di conda (verificare). Sarà sincronizzato anche visual studio (utilizzato per notebook jupiter) per disabilitare il venv

`deactivate` o `conda deactivate`

run venv, sulla root del progetto

`source venv/bin/activate`

attivare tensorboard da terminale

`tensroboard --logdir logs`

Per colab

```
%load_ext tensorboard # carico tensorboard su colab
%tensorboard --logdir logs
```

Runnare il venv

# Setup docker

Per salvare le dipendenze del virtual environment

`pip freeze > requirements.txt`

Nella root (buildare)

`docker build -t app .`

Run del container

`docker run app`