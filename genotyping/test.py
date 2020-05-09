import os
import sys


#print os.path.abspath(sys.argv[1])

os.chdir(os.path.dirname(os.path.abspath(sys.argv[1])))
print os.getcwd()

