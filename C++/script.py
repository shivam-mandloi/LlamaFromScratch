import os
import subprocess

# g++ -std=c++17 main.cpp -o main

class RunScript:
    def __init__(self, args, filename=""):
        self.wkDir = os.getcwd()
        self.filename = os.path.join(self.wkDir, filename)
        self.args = ["g++", "-std=c++23", self.filename] + self.GetAllDir() + ["-o", "main"] + args
        self.startRun = [os.path.join(self.wkDir,"./main")]

    def GetAllDir(self):
        dir = []
        includeDir = [self.wkDir]
        while(len(includeDir)):
            folder = includeDir.pop()
            for roots, dire, files in os.walk(folder):
                dir.append(roots)
        dir = ["-I" + i for i in dir]
        return dir

    def PrintStatus(self, result):
        if result.returncode == 0:
            print(f"[*]Compilation successful!\nFile Name: {self.filename}")
        else:
            print("[#]Compilation failed")
            print("Error message:")
            print(result.stderr.decode())
            print(self.args)

    def Run(self):        
        result = subprocess.run(self.args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.PrintStatus(result)

if __name__ == "__main__":
    filename = "main.cpp"
    script = RunScript([], filename)
    script.Run()
