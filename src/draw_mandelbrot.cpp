#include "draw_mandelbrot.hpp"
namespace Mandelbrot {
    void set_diff(complexBoundary& point) {
        point.x_diff = point.x_max - point.x_min;
        point.y_diff = point.y_max - point.y_min;
    }

    complexPoint map_pxl_to_complex(pixelLos pos, int height, int width, complexBoundary boundary) {
        return complexPoint{
            .x = boundary.x_min + (pos.x/width) * boundary.x_diff,
            .y = boundary.y_min + (pos.y/height) * boundary.y_diff
        };
    }

    void set_complex_boundary_zoom(complexBoundary& boundary, complexPoint& mouse_pos , double zoom) {
        set_diff(boundary);
        boundary.x_max = mouse_pos.x + (boundary.x_max - mouse_pos.x) / zoom;
        boundary.x_min = mouse_pos.x + (boundary.x_min - mouse_pos.x) / zoom;
        boundary.y_max = mouse_pos.y + (boundary.y_max - mouse_pos.y) / zoom;
        boundary.y_min = mouse_pos.y + (boundary.y_min - mouse_pos.y) / zoom;
    }

    sf::Vertex* render_mandelbrot(size_t height, size_t width, complexBoundary boundary) {
        static sf::Vertex* vertex_arr_host = nullptr;
        if (vertex_arr_host == nullptr) {
            vertex_arr_host = static_cast<sf::Vertex*>(malloc(sizeof(sf::Vertex) * width * height));
        }
        static sf::Vertex* vertex_arr_device = nullptr;
        if (vertex_arr_device == nullptr) {
            cudaMalloc((void**)vertex_arr_device, (width * height * sizeof(sf::Vertex)));
        }
        //kernel function call
        cudaMemcpy(vertex_arr_host, vertex_arr_device, (width * height * sizeof(sf::Vertex)), cudaMemcpyDeviceToHost);
        return vertex_arr_host;
    }
}