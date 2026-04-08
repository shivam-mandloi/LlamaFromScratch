import os
import subprocess

class RunScript:
    def __init__(self, args, filename="main.cpp"):
        self.wkDir = os.getcwd()
        self.filename = os.path.join(self.wkDir, filename)
        
        # 1. Define where LibTorch lives (MAKE SURE THIS PATH IS CORRECT)
        self.libtorch_path = "/mnt/c/Users/shiva/Desktop/IISC/LLAMA/A10/libtorch" 
        
        # 2. Add LibTorch Include Directories
        includes = [
            f"-I{self.libtorch_path}/include",
            f"-I{self.libtorch_path}/include/torch/csrc/api/include"
        ]
        
        # 3. Add LibTorch Library Directories and Runtime Path (rpath)
        lib_dirs = [
            f"-L{self.libtorch_path}/lib",
            f"-Wl,-rpath,{self.libtorch_path}/lib"
        ]
        
        # 4. FIX: Force the linker to keep the CUDA registries!
        libs = [
            "-Wl,--no-as-needed",
            "-ltorch", "-ltorch_cpu", "-ltorch_cuda", "-lc10", "-lc10_cuda",
            "-Wl,--as-needed",
            "-lcudart" # (Optional but recommended) explicitly link CUDA runtime
        ]
        
        # 5. Define the C++ ABI Flag
        abi_flag = ["-D_GLIBCXX_USE_CXX11_ABI=1"]

        # Assemble the final g++ command
        self.args = (
            ["g++", "-std=c++17", self.filename] 
            + abi_flag 
            + includes 
            + lib_dirs 
            + libs 
            + self.GetAllDir() 
            + ["-o", "main"] 
            + args
        )
        
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
            print(f"[*] Compilation successful!\nFile Name: {self.filename}")
            print("-" * 40)
            subprocess.run(["./main"])
        else:
            print("[#] Compilation failed")
            print("Error message:")
            print(result.stderr.decode())
            print("Command used:", " ".join(self.args))

    def Run(self):        
        print(f"Compiling your stupid code...")
        result = subprocess.run(self.args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.PrintStatus(result)

if __name__ == "__main__":
    script = RunScript([])
    script.Run()