#include <cuda_runtime.h>
#include <iostream>
#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include "mandelbrot.cuh"

struct c_planePos {
    double x;
    double y;
};

struct c_planePos mapToComplexPlane(double xc_max, double xc_min, double yc_max, double yc_min, double px, double py,
                                    double width, double height) {
    struct c_planePos c_pos{0};
    c_pos.x = xc_min + (px / width) * (xc_max - xc_min);
    c_pos.y = yc_min + (py / height) * (yc_max - yc_min);
    return c_pos;
}

void getNewRange(double &xc_max, double &xc_min, double &yc_max, double &yc_min, double zoom, struct c_planePos point) {
    xc_max = point.x + (xc_max - point.x) / zoom;
    xc_min = point.x + (xc_min - point.x) / zoom;
    yc_max = point.y + (yc_max - point.y) / zoom;
    yc_min = point.y + (yc_min - point.y) / zoom;
}

int main() {
    int count;
    constexpr int screen_width = 1600;
    constexpr int screen_height = 800;
    sf::RenderWindow window{sf::VideoMode({screen_width, screen_height}), "mandelbrot"};
    sf::Vertex *vertices_device;
    sf::Vertex *vertices_host = (sf::Vertex *) malloc(screen_width * screen_height * sizeof(sf::Vertex));
    cudaMalloc((void **) &vertices_device, (screen_width * screen_height * sizeof(sf::Vertex)));

    constexpr int iterations = 300;
    double x_re_max = 0.47;
    double x_re_min = -2.00;
    double y_im_max = 1.12;
    double y_im_min = -1.12;

    launch_mandelbrot(vertices_device, screen_width,
                      screen_height, iterations, x_re_max, x_re_min, y_im_max, y_im_min, 0, 0, 1);

    cudaMemcpy((void *) vertices_host, (void *) vertices_device, screen_width * screen_height * sizeof(sf::Vertex),
               cudaMemcpyDeviceToHost);
    cudaError err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "cuda error: " << cudaGetErrorString(err) << "\n";
    }
    // for (int i = 0; i < screen_width*screen_height; i++) {
    //     // std::cout << "pos x: " << vertices_host[i].position.x << "\n";
    //     // std::cout << "pos y: " << vertices_host[i].position.y << "\n";
    //     if (vertices_host[i].color == sf::Color::Blue)
    //         std::cout << "color os blue\n";
    // }
    while (window.isOpen()) {
        while (const std::optional event = window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                window.close();
            else if (const auto *scroll = event->getIf<sf::Event::MouseWheelScrolled>()) {
                double zoom_level = 1;
                if (scroll->delta > 0) {
                    zoom_level = 1.2;
                } else if (scroll->delta < 0) {
                    zoom_level = zoom_level / 1.2;
                }
                struct c_planePos complex_point = mapToComplexPlane(x_re_max, x_re_min, y_im_max, y_im_min,
                                                                    static_cast<double>(scroll->position.x),
                                                                    static_cast<double>(scroll->position.y),
                                                                    screen_width, screen_height);
                getNewRange(x_re_max, x_re_min, y_im_max, y_im_min, zoom_level, complex_point);
                launch_mandelbrot(vertices_device, screen_width,
                                  screen_height, iterations, x_re_max, x_re_min, y_im_max, y_im_min, scroll->position.x,
                                  scroll->position.y, zoom_level);

                cudaMemcpy((void *) vertices_host, (void *) vertices_device,
                           screen_width * screen_height * sizeof(sf::Vertex), cudaMemcpyDeviceToHost);
            }
            else if (event->is<sf::Event::MouseLeft>()) {

            }
        }
        window.clear();
        window.draw(vertices_host, screen_width * screen_height, sf::PrimitiveType::Points);
        window.display();
    }
}
