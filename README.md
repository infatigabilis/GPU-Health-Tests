# GPU Health Tests

## Install

```bash
virtualenv --system-site-packages -p python3 .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install --upgrade -r requirements.txt
```

## Run

```bash
.venv/bin/python pytorch-test.py
```

Test is passed if the script completes successfully and contained the "Test passed" output.

## Run TensorBoard (to see training progress)

```bash
tensorboard --logdir logs
```

---

## Install prerequisites (Ubuntu)

```bash
sudo apt install -y python3-pip
sudo pip3 install -U virtualenv
```
