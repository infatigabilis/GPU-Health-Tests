# GPUs Load Testing

## Install

```bash
virtualenv --system-site-packages -p python3 .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --upgrade -r requirements.txt
```

## Run

```bash
nohup .venv/bin/python pytorch-load-tester.py > training.out 2>&1 &
```

## Run TensorBoard (to see training progress)

```bash
nohup tensorboard --logdir logs > tensorboard.out 2>&1 &
```

---

## Install prerequisites (Ubuntu)

```bash
sudo apt install -y python3-pip
sudo pip3 install -U virtualenv
```
