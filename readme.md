
Activate venv, install requirements, then run:

- (The project was made with python 3.8 64 bit version in 2023, but most packages can safely be upgraded to newer versions)
    - (Example is the gym package, which has been replaced fully with gymnasium.)

```
python -m DDPG.visualize_ddpg
```

IF you dont have CUDA enabled / don't have a NVIDIA graphics card, you need to go into /DDPG/ddpg_torch.py, and set:

```
disable_cuda=True
```
