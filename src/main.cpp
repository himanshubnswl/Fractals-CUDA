#include <cuda_runtime.h>
#include <iostream>
#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include "mandelbrot.cuh"
#include "draw_mandelbrot.cuh"
#include "../cmake-build-debug-visual-studio/_deps/sfml-src/src/SFML/Window/InputImpl.hpp"
#include <chrono>
#include <thread>

int main() {
    constexpr int height = 800;
    constexpr int width = 1500;
    constexpr int iterations = 100;
    Mandelbrot::complexBoundary boundary{.x_max = 0.85, .x_min = -2.0, .y_max = 1.00, .y_min = -1.00};
    // Mandelbrot::complexBoundary boundary {0.85, -2.0, 0.8, -0.8};
    sf::RenderWindow window{sf::VideoMode{sf::Vector2u{width, height}}, "mandelbrot"};
    sf::Vertex *vertices = Mandelbrot::render_mandelbrot(height, width, boundary, iterations);
    // for (int i = 0; i < height * width; i++) {
    //     std::cout << "\nx is: " << vertices[i].position.x;
    //     std::cout << "\ny is: " << vertices[i].position.y;
    // }
    while (window.isOpen()) {
        static bool mouse_button_pressed = false;
        static sf::Vector2i oldpos{};
        static sf::Vector2i newpos{};
        // window.clear(sf::Color::Black);
        static bool redraw{false};
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>()) {
                window.close();
            } else if (const auto *scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
                double zoom{0};
                if (scroll->delta > 0) {
                    zoom = 1.2;
                } else {
                    zoom = 1 / 1.2;
                }
                Mandelbrot::pixelLos mouse_pos{
                    static_cast<double>(scroll->position.x), static_cast<double>(scroll->position.y)
                };
                auto mouse_com = Mandelbrot::map_pxl_to_complex(mouse_pos, height, width, boundary);
                Mandelbrot::set_complex_boundary_zoom(boundary, mouse_com, zoom);
                Mandelbrot::render_mandelbrot(height, width, boundary, iterations);
            } else if (auto *mouse = event->getIf<sf::Event::MouseButtonPressed>()) {
                mouse_button_pressed = true;
                oldpos = mouse->position;
            } else if (event->is<sf::Event::MouseMoved>() && mouse_button_pressed) {
                newpos = sf::Mouse::getPosition(window);
                sf::Vector2<double> delta = {
                    static_cast<double>(oldpos.x - newpos.x), static_cast<double>(oldpos.y - newpos.y)
                };
                if (delta.x != 0 || delta.y != 0) {
                    Mandelbrot::set_complex_boundary_drag(boundary, height, width, delta);
                    oldpos = newpos;
                    redraw = true;
                }
            } else if (event->is<sf::Event::MouseButtonReleased>()) {
                mouse_button_pressed = false;
            }
        }
        if (redraw) {
            Mandelbrot::render_mandelbrot(height, width, boundary, iterations);
            redraw = false;
        }
        window.draw(vertices, (height * width), sf::PrimitiveType::Points);
        window.display();
    }
}
