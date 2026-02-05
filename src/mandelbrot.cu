//
// Created by lhbdawn on 21-01-2026.
//
#include "draw_mandelbrot.hpp"

template<typename T>
__global__ void cal_color(sf::Vertex* const vertices, const T height, const T width, const int total_iterations,
                          const Mandelbrot::complexBoundary boundary) {
    T px = blockIdx.x * blockDim.x + threadIdx.x;
    T py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px > width || py > height) {
        return;
    }

    unsigned int idx = py * width * px;
    vertices[idx].position.x = px;
    vertices[idx].position.y = py;
    const Mandelbrot::complexPoint c_const{
        .x = boundary.x_min + (px / width) * boundary.x_diff,
        .y = boundary.y_min + (py / height) * boundary.y_diff
    };
    Mandelbrot::complexPoint xy{
        .x = 0.0,
        .y = 0.0
    };
    unsigned int current_iteration = 0;
    T c_magnitude = xy.x * xy.x + xy.y * xy.y;
    T x_temp = 0;
    while (c_magnitude < 4 && current_iteration != total_iterations) {
        x_temp = xy.x * xy.x - xy.y * xy.y + c_const.x;
        xy.y = 2 * xy.x * xy.y + c_const.y;
        xy.x = x_temp;
        c_magnitude = xy.x * xy.x + xy.y * xy.y;
        current_iteration++;
    }

    if (current_iteration == total_iterations) {
        vertices[idx].color = sf::Color::Black;
    } else {
        vertices[idx].color = sf::;
    }
}

void launch_mandelbrot(sf::Vertex *vertices, int width, int height, int total_iterations, double x_scale_max,
                       double x_scale_min, double y_scale_max, double y_scale_min, double x_offset, double y_offset,
                       double zoom_level) {
    dim3 blockSize(32, 32);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    draw_mandelbrot<double><<<gridSize, blockSize>>>(
        vertices, (double) height, (double) width, total_iterations, x_scale_max, x_scale_min, y_scale_max,
        y_scale_min, x_offset, y_offset, zoom_level);
    cudaDeviceSynchronize(); // Ensure it finishes before returning to main.cpp
}
