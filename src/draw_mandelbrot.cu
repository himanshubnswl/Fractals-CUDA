#include "draw_mandelbrot.cuh"

// __device__ double Mandelbrot::HSL::HuetoRGB(double arg1, double arg2, double H) {
//     if ( H < 0 ) H += 1;
//     if ( H > 1 ) H -= 1;
//     if ( ( 6 * H ) < 1 ) { return (arg1 + ( arg2 - arg1 ) * 6 * H); }
//     if ( ( 2 * H ) < 1 ) { return arg2; }
//     if ( ( 3 * H ) < 2 ) { return ( arg1 + ( arg2 - arg1 ) * ( ( 2.0 / 3.0 ) - H ) * 6 ); }
//     return arg1;
// }
// __device__ void Mandelbrot::HSL::HSLtoRGB(sf::Color& target) {
//     double H = hue/360.0;
//     double S = saturation/100.0;
//     double L = luminance/100.0;
//     constexpr double D_EPSILON = 0.00000000000001;
//
//
//     if (S <= D_EPSILON)
//     {
//         target.r = target.g = target.b = 255;
//     }
//     else {
//         double arg1, arg2;
//         if ( L < 0.5 ) { arg2 = L * ( 1 + S ); }
//         else { arg2 = ( L + S ) - ( S * L ); }
//         arg1 = 2 * L - arg2;
//
//         target.r = ( 255 * HSL::HuetoRGB( arg1, arg2, (H + 1.0/3.0 ) ) );
//         target.g = ( 255 * HSL::HuetoRGB( arg1, arg2, H ) );
//         target.b = ( 255 * HSL::HuetoRGB( arg1, arg2, (H - 1.0/3.0 ) ) );
//         target.a = 255;
//     }
// }

namespace Mandelbrot {
    void set_diff(complexBoundary& point) {
        point.x_diff = point.x_max - point.x_min;
        point.y_diff = point.y_max - point.y_min;
    }

    complexPoint map_pxl_to_complex(pixelLos pos, int height, int width, complexBoundary boundary) {
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

    sf::Vertex* render_mandelbrot(size_t height, size_t width, complexBoundary boundary) {
        static sf::Vertex* vertex_arr_host = nullptr;
        set_diff(boundary);
        if (vertex_arr_host == nullptr) {
            vertex_arr_host = static_cast<sf::Vertex*>(malloc(sizeof(sf::Vertex) * width * height));
        }
        static sf::Vertex* vertex_arr_device = nullptr;
        if (vertex_arr_device == nullptr) {
            cudaMalloc((void**)&vertex_arr_device, (width * height * sizeof(sf::Vertex)));
        }
        launch_mandelbrot_kernel(vertex_arr_device, height, width, 1000, boundary);
        cudaMemcpy(vertex_arr_host, vertex_arr_device, (width * height * sizeof(sf::Vertex)), cudaMemcpyDeviceToHost);
        return vertex_arr_host;
    }
}