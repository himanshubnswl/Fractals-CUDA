#pragma once
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include "mandelbrot.cuh"
#include <cmath>

namespace Mandelbrot {
    struct complexPoint {
        double x;
        double y;
    };

    struct complexBoundary {
        double x_max;
        double x_min;
        double x_diff;
        double y_max;
        double y_min;
        double y_diff;

        // complexBoundary(double x_max, double x_min, double y_max, double y_min) : x_max(x_max), x_min(x_min),
            // y_max(y_max), y_min(y_min) {}
    };

    struct HSL {
        double hue;
        double saturation;
        double luminance;

        __host__ __device__ double HuetoRGB(double arg1, double arg2, double H);

        __host__ __device__ sf::Color HSLtoRGB();
    };

    using pixelLos = complexPoint;

    void set_diff(complexBoundary &point);

    complexPoint map_pxl_to_complex(pixelLos pos, int height, int width, complexBoundary boundary);

    void set_complex_boundary_zoom(complexBoundary &boundary, complexPoint &mouse_pos, double zoom);

    sf::Vertex *render_mandelbrot(size_t height, size_t width, complexBoundary boundary);
}
