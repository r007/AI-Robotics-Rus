# Установка окружения

### Среда выполнения
```shell
$ mamba create -n ai-hardware python=3.8
$ mamba activate ai-hardware
$ pip install setuptools==65.5.0 pip==21 wheel==0.38.0
$ pip install gym==0.21.0
$ pip install matplotlib==3.7.5
$ pip install pyglet==1.4.0
$ pip install scipy==1.10.1
$ pip install casadi==3.7.2
```

### Алгоритм PID
```shell
$ python cartpole-PID.py
```

### Алгоритм MPC
```shell
$ python cartpole_MPC.py
```

### Алгоритм LQR
```shell
$ python cartpole-LQR.py
```
