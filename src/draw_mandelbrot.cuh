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

        inline __device__ double HuetoRGB(double arg1, double arg2, double H) {
            if (H < 0) H += 1;
            if (H > 1) H -= 1;
            if ((6 * H) < 1) { return (arg1 + (arg2 - arg1) * 6 * H); }
            if ((2 * H) < 1) { return arg2; }
            if ((3 * H) < 2) { return (arg1 + (arg2 - arg1) * ((2.0 / 3.0) - H) * 6); }
            return arg1;
        }

        inline __device__ void HSLtoRGB(sf::Color &target) {
            double H = hue / 360.0;
            double S = saturation / 100.0;
            double L = luminance / 100.0;
            constexpr double D_EPSILON = 0.00000000000001;


            if (S <= D_EPSILON) {
                target.r = target.g = target.b = 255;
            } else {
                double arg1, arg2;
                if (L < 0.5) { arg2 = L * (1 + S); } else { arg2 = (L + S) - (S * L); }
                arg1 = 2 * L - arg2;

                target.r = (255 * HSL::HuetoRGB(arg1, arg2, (H + 1.0 / 3.0)));
                target.g = (255 * HSL::HuetoRGB(arg1, arg2, H));
                target.b = (255 * HSL::HuetoRGB(arg1, arg2, (H - 1.0 / 3.0)));
                target.a = 255;
            }
        }
    };

    using pixelLos = complexPoint;

    void set_diff(complexBoundary &point);

    complexPoint map_pxl_to_complex(sf::Vector2i pos, int height, int width, complexBoundary boundary);

    void set_complex_boundary_zoom(complexBoundary &boundary, complexPoint mouse_pos, double zoom);

    sf::Vertex *render_mandelbrot(int height, int width, complexBoundary boundary, int iterations);

    sf::Vertex *render_julia(int height, int width, complexBoundary boundary, int max_iterations,
                             complexPoint constant_p);

    __host__ void set_complex_boundary_drag(complexBoundary &boundary, int height, int width,
                                            sf::Vector2<double> delta);
}
