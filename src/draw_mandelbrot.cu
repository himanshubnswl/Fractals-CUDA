#include "draw_mandelbrot.cuh"
#include <iostream>

namespace Mandelbrot {
    void set_diff(complexBoundary& point) {
        point.x_diff = point.x_max - point.x_min;
        point.y_diff = point.y_max - point.y_min;
    }

    complexPoint map_pxl_to_complex(sf::Vector2i pos, int height, int width, complexBoundary boundary) {
        return complexPoint{
            .x = boundary.x_min + (static_cast<double>(pos.x)/width) * boundary.x_diff,
            .y = boundary.y_min + (static_cast<double>(pos.y)/height) * boundary.y_diff
        };
    }

    void set_complex_boundary_zoom(complexBoundary& boundary, complexPoint& mouse_pos , double zoom) {
        set_diff(boundary);
        boundary.x_max = mouse_pos.x + (boundary.x_max - mouse_pos.x) / zoom;
        boundary.x_min = mouse_pos.x + (boundary.x_min - mouse_pos.x) / zoom;
        boundary.y_max = mouse_pos.y + (boundary.y_max - mouse_pos.y) / zoom;
        boundary.y_min = mouse_pos.y + (boundary.y_min - mouse_pos.y) / zoom;
    }

    sf::Vertex* render_mandelbrot(int height, int width, complexBoundary boundary, int iterations) {
        static sf::Vertex* vertex_arr_host = nullptr;
        set_diff(boundary);
        if (vertex_arr_host == nullptr) {
            vertex_arr_host = static_cast<sf::Vertex*>(malloc(sizeof(sf::Vertex) * width * height));
        }
        static sf::Vertex* vertex_arr_device = nullptr;
        if (vertex_arr_device == nullptr) {
            cudaMalloc((void**)&vertex_arr_device, (width * height * sizeof(sf::Vertex)));
        }
        launch_mandelbrot_kernel(vertex_arr_device, height, width, iterations, boundary);
        cudaMemcpy(vertex_arr_host, vertex_arr_device, (width * height * sizeof(sf::Vertex)), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < height * width; i++) {
        //     std::cout << "\nx is: " << vertex_arr_host[i].position.x;
        //     std::cout << "\ny is: " << vertex_arr_host[i].position.y;
        // }
        return vertex_arr_host;
    }

    sf::Vertex* render_julia(int height, int width, complexBoundary boundary, int max_iterations, complexPoint constant_p) {
        static sf::Vertex* vertex_arr_host = nullptr;
        set_diff(boundary);
        if (vertex_arr_host == nullptr) {
            vertex_arr_host = static_cast<sf::Vertex*>(malloc(sizeof(sf::Vertex) * width * height));
        }
        static sf::Vertex* vertex_arr_device = nullptr;
        if (vertex_arr_device == nullptr) {
            cudaMalloc((void**)&vertex_arr_device, (width * height * sizeof(sf::Vertex)));
        }
        launch_julia_kernel(vertex_arr_device, height, width, max_iterations, constant_p);
        cudaMemcpy(vertex_arr_host, vertex_arr_device, (width * height * sizeof(sf::Vertex)), cudaMemcpyDeviceToHost);
        // for (int i = 0; i < height * width; i++) {
        //     std::cout << "\nx is: " << vertex_arr_host[i].position.x;
        //     std::cout << "\ny is: " << vertex_arr_host[i].position.y;
        // }
        return vertex_arr_host;
    }

    __host__ void set_complex_boundary_drag(complexBoundary& boundary,int height, int width, sf::Vector2<double> delta) {
        delta.x = (delta.x/width) * boundary.x_diff;
        delta.y = (delta.y/height) * boundary.y_diff;
        boundary.x_max += delta.x;
        boundary.x_min += delta.x;
        boundary.y_max += delta.y;
        boundary.y_min += delta.y;
    }
}