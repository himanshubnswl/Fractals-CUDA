//
// Created by lhbdawn on 24-01-2026.
//

#ifndef MANDELBROT_CUH
#define MANDELBROT_CUH
#include <SFML/Graphics.hpp>
template<typename T>
void draw_mandelbrot(sf::Vertex* vertices ,T height, T width, int total_iterations, T x_scale_max, T x_scale_min, T y_scale_max, T y_scale_min);
void launch_mandelbrot(sf::Vertex *vertices, int width, int height, int total_iterations, double x_scale_max,
                       double x_scale_min, double y_scale_max, double y_scale_min, double x_offset, double y_offset, double zoom_level);
#endif //MANDELBROT_CUH
