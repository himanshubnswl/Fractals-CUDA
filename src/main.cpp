#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include "draw_mandelbrot.cuh"

bool event_handler(sf::Window &window, Mandelbrot::complexBoundary &boundary, int height, int width) {
    static bool mouse_button_pressed{false};
    static sf::Vector2i oldpos{};
    static sf::Vector2i newpos{};

    bool redraw{false};

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
            Mandelbrot::set_complex_boundary_zoom(
                boundary, Mandelbrot::map_pxl_to_complex(scroll->position, height, width, boundary), zoom);
            redraw = true;
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
    return redraw;
}


int main() {
    constexpr int height = 800;
    constexpr int width = 1500;
    constexpr unsigned int iterations_mandelbrot = 300;
    constexpr unsigned int iterations_julia = 300;
    Mandelbrot::complexBoundary boundary_mandelbrot{.x_max = 0.85, .x_min = -2.0, .y_max = 1.00, .y_min = -1.00};
    Mandelbrot::complexBoundary boundary_julia{.x_max = 0.85, .x_min = -2.0, .y_max = 1.00, .y_min = -1.00};
    // Mandelbrot::complexBoundary boundary {0.85, -2.0, 0.8, -0.8};
    sf::RenderWindow window_mandelbrot{sf::VideoMode{sf::Vector2u{width, height}}, "mandelbrot"};
    sf::RenderWindow window_julia{
        sf::VideoMode{sf::Vector2u{width, height}},
        "julia"
    };
    sf::Vertex *vertices_mandelbrot = Mandelbrot::render_mandelbrot(height, width, boundary_mandelbrot,
                                                                    iterations_mandelbrot);
    sf::Vertex *vertices_julia = Mandelbrot::render_julia(height, width, boundary_julia, iterations_julia,
                                                          Mandelbrot::map_pxl_to_complex(
                                                              sf::Mouse::getPosition(window_mandelbrot), height, width,
                                                              boundary_julia));
    // for (int i = 0; i < height * width; i++) {
    //     std::cout << "\nx is: " << vertices[i].position.x;
    //     std::cout << "\ny is: " << vertices[i].position.y;
    // }
    while (window_mandelbrot.isOpen()) {
        // static bool mouse_button_pressed = false;
        // static sf::Vector2i oldpos{};
        // static sf::Vector2i newpos{};
        // // window.clear(sf::Color::Black);
        // static bool redraw_mandelbrot{false};
        // //event loop
        // while (const std::optional event = window_mandelbrot.pollEvent()) {
        //     if (event->is<sf::Event::Closed>()) {
        //         window_mandelbrot.close();
        //     } else if (const auto *scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
        //         double zoom{0};
        //         if (scroll->delta > 0) {
        //             zoom = 1.2;
        //         } else {
        //             zoom = 1 / 1.2;
        //         }
        //         auto mouse_com = Mandelbrot::map_pxl_to_complex(scroll->position, height, width, boundary_mandelbrot);
        //         Mandelbrot::set_complex_boundary_zoom(boundary_mandelbrot, mouse_com, zoom);
        //         redraw_mandelbrot = true;
        //     } else if (auto *mouse = event->getIf<sf::Event::MouseButtonPressed>()) {
        //         if (mouse->button == sf::Mouse::Button::Left) {
        //             mouse_button_pressed = true;
        //             oldpos = mouse->position;
        //         } else if (mouse->button == sf::Mouse::Button::Right) {
        //             vertices_mandelbrot = Mandelbrot::render_julia(height, width, boundary_mandelbrot, 300,
        //                                                            Mandelbrot::map_pxl_to_complex(
        //                                                                mouse->position, height, width,
        //                                                                boundary_mandelbrot));
        //         }
        //     } else if (event->is<sf::Event::MouseMoved>() && mouse_button_pressed) {
        //         newpos = sf::Mouse::getPosition(window_mandelbrot);
        //         sf::Vector2<double> delta = {
        //             static_cast<double>(oldpos.x - newpos.x), static_cast<double>(oldpos.y - newpos.y)
        //         };
        //         if (delta.x != 0 || delta.y != 0) {
        //             Mandelbrot::set_complex_boundary_drag(boundary_mandelbrot, height, width, delta);
        //             oldpos = newpos;
        //             redraw_mandelbrot = true;
        //         }
        //     } else if (event->is<sf::Event::MouseButtonReleased>()) {
        //         mouse_button_pressed = false;
        //     }
        // }
        bool redraw_mandelbrot = event_handler(window_mandelbrot, boundary_mandelbrot, height, width);
        bool redraw_julia = event_handler(window_julia, boundary_julia, height, width);
        if (redraw_mandelbrot) {
            Mandelbrot::render_mandelbrot(height, width, boundary_mandelbrot, iterations_mandelbrot);
            redraw_mandelbrot = false;
        }
        if (redraw_julia) {
            Mandelbrot::render_julia(height, width, boundary_julia, iterations_julia,
                                     Mandelbrot::map_pxl_to_complex(sf::Mouse::getPosition(window_julia), height, width,
                                                                    boundary_julia));
            redraw_julia = false;
        }
        window_mandelbrot.draw(vertices_mandelbrot, (height * width), sf::PrimitiveType::Points);
        window_julia.draw(vertices_julia, (height * width), sf::PrimitiveType::Points);
        window_mandelbrot.display();
    }
}
