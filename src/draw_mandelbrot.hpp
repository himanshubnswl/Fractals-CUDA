#pragma once
#include <cuda_runtime.h>
#include <SFML/Graphics.hpp>
#include <SFML/Graphics/Color.hpp>
#include <cmath>
#include <algorithm>

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
    };
    struct HSL {
        double hue;
        double saturation;
        double luminance;

        double HuetoRGB(double arg1, double arg2, double H);
        sf::Color HSLtoRGB();
    };
    using pixelLos = complexPoint;

    void set_diff(complexBoundary& point);
    complexPoint map_pxl_to_complex(pixelLos pos, int height, int width, complexBoundary boundary);
    void set_complex_boundary_zoom(complexBoundary& boundary, complexPoint& mouse_pos , double zoom);
    sf::Vertex* render_mandelbrot(size_t height, size_t width, complexBoundary boundary);
}