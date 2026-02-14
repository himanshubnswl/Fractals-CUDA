//
// Created by lhbdawn on 24-01-2026.
//
#pragma once

#define MANDELBROT_CUH
#include <SFML/Graphics.hpp>

namespace Mandelbrot {
    struct complexBoundary;
    struct complexPoint;
}

void launch_mandelbrot_kernel(sf::Vertex *const vertices, const int height, const int width, const int total_iterations,
                              const Mandelbrot::complexBoundary boundary);

void launch_julia_kernel(sf::Vertex * const vertices, const int height, const int width, const int max_iterations,
 const Mandelbrot::complexPoint constant_p);
