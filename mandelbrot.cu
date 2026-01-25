//
// Created by lhbdawn on 21-01-2026.
//
#include <SFML/Graphics.hpp>
#include <iostream>

template<typename T>
__global__ void draw_mandelbrot(sf::Vertex* vertices ,T height, T width, int total_iterations, T x_scale_max, T x_scale_min, T y_scale_max, T y_scale_min) {
    T px = blockIdx.x * blockDim.x + threadIdx.x;
    T py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px > width || py > height) {
        return ;
    }
    unsigned int idx = (py * width + px);

    vertices[idx].position.x = px;
    vertices[idx].position.y = py;

    T c_real = x_scale_min + (px/width) * (x_scale_max - x_scale_min);
    T c_imag = y_scale_min + (py/height) * (y_scale_max - y_scale_min);

    T x = 0.0;
    T y = 0.0;
    unsigned int current_iteration = 0;
    T c_magnitude = x*x + y*y;
    T x_temp = 0;
    while (c_magnitude < 4 && current_iteration != total_iterations) {
        x_temp = x*x - y*y + c_real;
        y = 2*x*y + c_imag;
        x = x_temp;
        c_magnitude = x*x + y*y;
        current_iteration++;
    }
    if (current_iteration == total_iterations) {
        vertices[idx].color = sf::Color::Black;
    }
    else {
        vertices[idx].color = sf::Color::Blue;
    }
    return;
}

void launch_mandelbrot(sf::Vertex* vertices, int width, int height, int iterations) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    draw_mandelbrot<double><<<gridSize, blockSize>>>(
        vertices, (double)height, (double)width, iterations, 1.27, -2.00, 1.12, -1.12
    );
    cudaDeviceSynchronize(); // Ensure it finishes before returning to main.cpp
}