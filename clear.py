import shutil
import os

if __name__ == "__main__":
    shutil.rmtree("clients")
    shutil.rmtree("servers")
    os.remove("logs.txt")