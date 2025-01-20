def _load_extension():
    import torch
    import os

    lib_path = os.path.join(os.path.dirname(__file__), 'hpim.cpython-36m-x86_64-linux-gnu.so')
    torch.ops.load_library(lib_path)

    # torch.matmul = torch.ops.hpim.pim_matmul
    # torch.add = torch.ops.hpim.pim_add
    # torch.transpose = torch.ops.hpim.pim_transpose


_load_extension()

from .hpim_optimize import optimize

# co tutaj bedzie/powinno byc?
# ladowanie z torch.ops.load_library i moze jakas zmiana na ladniejsze nazwy z opisami i syntaxami
# wynik: np. hpim.matmul, tez hpim.optimize(model.eval(), operations=...) z pliku optimize.py
# ostatecznie paczka bedzie miala: init, optimize.py i jakies pliki .so
# hpim.optimize bedzie zamienialo forwardy w modulach rekurencyjnie ?