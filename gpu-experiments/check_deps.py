import sys
print("Python version:", sys.version)
try:
    import transformers
    print("transformers:", transformers.__version__)
except ImportError as e:
    print("transformers import error:", e)
try:
    import pandas
    print("pandas:", pandas.__version__)
except ImportError as e:
    print("pandas import error:", e)
try:
    import numpy
    print("numpy:", numpy.__version__)
except ImportError as e:
    print("numpy import error:", e)
try:
    import vllm
    print("vllm:", vllm.__version__)
except ImportError as e:
    print("vllm import error:", e)
