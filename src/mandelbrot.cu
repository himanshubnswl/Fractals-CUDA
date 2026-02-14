//
// Created by lhbdawn on 21-01-2026.
//
#include "draw_mandelbrot.cuh"
#include "mandelbrot.cuh"

template<typename T>
__global__ void cal_color(sf::Vertex *const vertices, const T height, const T width, const int total_iterations,
                          const Mandelbrot::complexBoundary boundary) {
    T px = blockIdx.x * blockDim.x + threadIdx.x;
    T py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px > width || py > height) {
        return;
    }

    unsigned int idx = py * width + px;
    vertices[idx].position.x = px;
    vertices[idx].position.y = py;
    const Mandelbrot::complexPoint c_const{
        .x = boundary.x_min + (static_cast<double>(px) / width) * boundary.x_diff,
        .y = boundary.y_min + (static_cast<double>(py) / height) * boundary.y_diff
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
    constexpr double LN_2 = 0.30102999;
    constexpr double HUE = 200;
    constexpr double SAT = 50.00;
    constexpr double LUM = 30.00;
    if (current_iteration == total_iterations) {
        vertices[idx].color = sf::Color::Black;
    } else {
        Mandelbrot::HSL color_hsl{};
        color_hsl.saturation = fmin(
            100.00, (log(static_cast<double>(current_iteration) + 1.00 - log(log(c_magnitude)) / LN_2) * 75.00));
        color_hsl.hue = fmod(HUE, 360.00);
        color_hsl.luminance = fmod(LUM, 100.00);
        color_hsl.HSLtoRGB(vertices[idx].color);
        // vertices[idx].color = sf::Color::Blue;
    }
}

__global__ void cal_color_julia(sf::Vertex *const vertices, const double height, const double width,
                                const int max_iterations, const Mandelbrot::complexBoundary boundary,
                                const Mandelbrot::complexPoint constant_p) {
    unsigned int px = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int py = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int idx = static_cast<int>(py * width + px);
    Mandelbrot::complexPoint xy{
        .x = boundary.x_min + (px / width) * boundary.x_diff, .y = boundary.y_min + (py / height) * boundary.y_diff
    };
    double magnitude = xy.x * xy.x + xy.y * xy.y;
    double x_temp = 0;
    unsigned int current_iteration = 0;
    while (magnitude < 4 && current_iteration < max_iterations) {
        x_temp = xy.x * xy.x - xy.y * xy.y + constant_p.x;
        xy.y = 2 * xy.x * xy.y + constant_p.y;
        xy.x = x_temp;
        magnitude = xy.x * xy.x + xy.y * xy.y;
        current_iteration++;
    }

    constexpr double LN_2 = 0.30102999;
    constexpr double HUE = 200;
    constexpr double SAT = 50.00;
    constexpr double LUM = 30.00;
    if (current_iteration == max_iterations) {
        vertices[idx].color = sf::Color::Black;
    } else {
        Mandelbrot::HSL color_hsl{};
        color_hsl.saturation = fmin(
            100.00, (log(static_cast<double>(current_iteration) + 1.00 - log(log(magnitude)) / LN_2) * 75.00));
        color_hsl.hue = fmod(HUE, 360.00);
        color_hsl.luminance = fmod(LUM, 100.00);
        color_hsl.HSLtoRGB(vertices[idx].color);
        // vertices[idx].color = sf::Color::Blue;
    }
}

void launch_julia_kernel(sf::Vertex *const vertices, const int height, const int width, const int max_iterations,
                         const Mandelbrot::complexBoundary boundary,
                         const Mandelbrot::complexPoint constant_p
) {
    dim3 blockSize{32, 32};
    dim3 gridSize{(width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y};
    cal_color_julia<<<gridSize,blockSize>>>(vertices, static_cast<double>(height), static_cast<double>(width),
                                            max_iterations,boundary, constant_p);
    cudaDeviceSynchronize();
}

void launch_mandelbrot_kernel(sf::Vertex *const vertices, const int height, const int width, const int total_iterations,
                              const Mandelbrot::complexBoundary boundary) {
    dim3 blockSize{32, 32};
    dim3 gridSize{(width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y};
    cal_color<double><<<gridSize,blockSize>>>(vertices, height, width, total_iterations, boundary);
    cudaDeviceSynchronize();
}
