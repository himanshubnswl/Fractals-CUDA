//
// Created by lhbdawn on 24-01-2026.
//

#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH
#include <SFML/Graphics.hpp>

void launch_mandelbrot_kernel(sf::Vertex *const vertices, const int height, const int width, const int total_iterations,
                              const Mandelbrot::complexBoundary boundary);

#endif //MANDELBROT_CUH
