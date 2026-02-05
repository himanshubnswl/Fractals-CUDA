#include "draw_mandelbrot.hpp"

double Mandelbrot::HSL::HuetoRGB(double arg1, double arg2, double H) {
    if ( H < 0 ) H += 1;
    if ( H > 1 ) H -= 1;
    if ( ( 6 * H ) < 1 ) { return (arg1 + ( arg2 - arg1 ) * 6 * H); }
    if ( ( 2 * H ) < 1 ) { return arg2; }
    if ( ( 3 * H ) < 2 ) { return ( arg1 + ( arg2 - arg1 ) * ( ( 2.0 / 3.0 ) - H ) * 6 ); }
    return arg1;
}
sf::Color Mandelbrot::HSL::HSLtoRGB() {
    double H = hue/360.0;
    double S = saturation/100.0;
    double L = luminance/100.0;
    constexpr double D_EPSILON = 0.00000000000001;
    double arg1, arg2;

    if (S <= D_EPSILON)
    {
        sf::Color C(L*255, L*255, L*255);
        return C;
    }
    else {
        if ( L < 0.5 ) { arg2 = L * ( 1 + S ); }
        else { arg2 = ( L + S ) - ( S * L ); }
        arg1 = 2 * L - arg2;

        sf::Uint8 r =( 255 * HSL::HuetoRGB( arg1, arg2, (H + 1.0/3.0 ) ) );
        sf::Uint8 g =( 255 * HSL::HuetoRGB( arg1, arg2, H ) );
        sf::Uint8 b =( 255 * HSL::HuetoRGB( arg1, arg2, (H - 1.0/3.0 ) ) );
        sf::Color C(r,g,b);
        return C;
    }
}

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