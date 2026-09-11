# omnert-tiny-125M

Latest Omnert v2 instruction model plus the minimal files needed to run inference.

## Install

```powershell
git lfs install
git clone https://github.com/Blu3yDev/omnert-tiny-125M.git
cd omnert-tiny-125M
py -3.12 -m pip install -r requirements.txt
```

## Run

```powershell
py -3.12 src/chat_v2.py
```

Optional:

```powershell
py -3.12 src/chat_v2.py --temperature 0.75 --top-p 0.9
```

Type `/quit` to exit.
