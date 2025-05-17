import sdv, os, pkgutil, sys

print("sys.executable:", sys.executable)
print("sys.path:", sys.path, "\n")

print("sdv module file:", sdv.__file__)
print("site-packages/sdv contents:", os.listdir(os.path.dirname(sdv.__file__)), "\n")

# ¿Contiene subcarpeta tabular?
tab_path = os.path.join(os.path.dirname(sdv.__file__), "tabular")
print("¿Existe sdv/tabular?", os.path.isdir(tab_path))
if os.path.isdir(tab_path):
    print("  → Contenido de sdv/tabular:", os.listdir(tab_path))
