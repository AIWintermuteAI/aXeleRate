import subprocess
import os

cwd = os.path.dirname(os.path.realpath(__file__))
#rc, out = subprocess.getstatusoutput('dpkg -l edgetpu-compiler')
#if rc != 0: print ("Error occurred:", out)
subprocess.Popen(['bash install_edge_tpu_compiler.sh'], shell=True, stdin=subprocess.PIPE, cwd=cwd).communicate()
