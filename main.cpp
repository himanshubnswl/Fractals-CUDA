#include <cuda_runtime.h>
#include <iostream>
#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include "mandelbrot.cuh"

int main() {
    constexpr int screen_width = 1600;
    constexpr int screen_height = 800;
    sf::RenderWindow window {sf::VideoMode({screen_width, screen_height}), "mandelbrot"};
    sf::Vertex* vertices_device;
    sf::Vertex* vertices_host = (sf::Vertex*)malloc(screen_width * screen_height * sizeof(sf::Vertex));
    cudaMalloc((void**)&vertices_device, (screen_width * screen_height * sizeof(sf::Vertex)));

    launch_mandelbrot(vertices_device, screen_width,
        screen_height, 300);

    cudaMemcpy((void*)vertices_host, (void*)vertices_device, screen_width*screen_height*sizeof(sf::Vertex), cudaMemcpyDeviceToHost);
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
        }
        window.clear();
        window.draw(vertices_host, screen_width * screen_height, sf::PrimitiveType::Points);
        window.display();
    }
}