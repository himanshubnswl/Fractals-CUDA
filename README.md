Welcome to fractals, a visualization of the mandelbrot set and it's corresponding julia set.
features are zooming, dragging to interact with the set.

Uses nvidia'a CUDA for parallel processing of the set, making it much faster than CPU implementations of the same.

How to run?
Windows: You need Microsoft's visual studio compiler as CUDA compiler will only use it on windows, you need to activate the script located at
```C:\Program Files\Microsoft Visual Studio\18\Professional\VC\Auxiliary\Build\vcvarsall.bat```

next you need to pass in arguments to the script for your architecture eg ```amd64``` and also specify the version of msvc to use, do note CUDA compiler needs 
msvc version 14.44 or previous

You also need to have the CUDA toolkit ```https://developer.nvidia.com/cuda/toolkit``` <--- link
finally run
```cmake -B ./build```
```cd build```
```make```
and it should compile

For Linux: idk i haven't compiled it on linux but CUDA works with gcc and clang so it should be easier than doing it on windows


<img width="1920" height="1080" alt="26--21-09-37-414" src="https://github.com/user-attachments/assets/d08bc870-4ef5-4df6-a866-ac1e11e6ae1c" />
