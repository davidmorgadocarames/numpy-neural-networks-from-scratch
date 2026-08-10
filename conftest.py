"""Hace importables `capas.py` (raíz) y `capas_cnn.py` (08-cnn-fashion-mnist/) desde
tests/, sin convertir ninguna carpeta del repo en un paquete Python instalable."""

import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "08-cnn-fashion-mnist"))
