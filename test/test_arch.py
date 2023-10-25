import sys,pytest
sys.path.append(".")

from src.nn import arch

from src import core
import json


with open("test/GLOBAL_CONFIG.txt","w") as file:
    file.write(str(core.GLOBAL_CONFIG))
