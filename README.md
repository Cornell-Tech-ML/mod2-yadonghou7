[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/YFgwt0yY)
# MiniTorch Module 2

<img src="https://minitorch.github.io/minitorch.svg" width="50%">


* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module2/module2/

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py project/run_manual.py project/run_scalar.py project/datasets.py



1. Simple Model <br>
PTS = 50 <br>
DATASET = minitorch.datasets["Simple"](PTS)
HIDDEN = 2 <br>
RATE = 0.5 <br>
Number of eoichs: 400 <br>

![simple](./image/Screenshot1.png)

2. Diag Model <br>
PTS = 100 <br>
DATASET = minitorch.datasets["Diag"](PTS)
HIDDEN = 5 <br>
RATE = 0.1 <br>
Number of eoichs: 1500 <br>

![Diag](./image/Screenshot2.png)


3. Split Model <br>
PTS = 50 <br>
DATASET = minitorch.datasets["Split"](PTS)
HIDDEN = 6 <br>
RATE = 0.1  <br>
Number of eoichs: 15 <br>

![Split](./image/Screenshot3.png)

4. XOR Model <br>
PTS = 50 <br>
DATASET = minitorch.datasets["XOR"](PTS)
HIDDEN = 6 <br>
RATE = 0.5 <br>
Number of eoichs: 1225 <br>

![XOR](./image/Screenshot4.png)


