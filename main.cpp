#include <SFML/Audio.hpp>
#include <SFML/Graphics.hpp>
#include <iostream>

int main()
{
    constexpr float height {1000.00f};
    constexpr float width {1700.00f};
    sf::RenderWindow main_window(sf::VideoMode({static_cast<unsigned int>(width),static_cast<unsigned int>(height)}), "main_window");
    sf::Vertex point{{200.0f, 200.0f}, sf::Color::Red};
    sf::VertexArray vertices {sf::PrimitiveType::Points, static_cast<unsigned int>(width*height)};
    int vertices_idx {0};
    for (float i = 0.0f; i < width; i++) {
        for (float j = 0.0f; j < height; j++) {
            vertices[vertices_idx].position = {i,j};
            vertices_idx++;
        }
    }

    constexpr float X_scaleMax = 1.27f;
    constexpr float X_scaleMin = -2.00f;
    constexpr float Y_scaleMax = 1.12f;
    constexpr float Y_scaleMin = -1.12f;

    constexpr int max_iteration = 1000;
    for (int i = 0; i < width*height; i++) {
        float x_scaled = X_scaleMin + (vertices[i].position.x/width) * (X_scaleMax - X_scaleMin);
        float y_scaled = Y_scaleMin + (vertices[i].position.y/height) * (Y_scaleMax - Y_scaleMin);
        float x = 0.0f;
        float y = 0.0f;
        unsigned int iteration = 0;
        while (x*x + y*y <= 4 && iteration < max_iteration) {
            float x_temp = x*x - y*y + x_scaled;
            y = 2*x*y + y_scaled;
            x = x_temp;
            iteration++;
        }
        if (iteration == max_iteration) {
            vertices[i].color = sf::Color::Black;
        }
        else {
            // std::uint8_t brightness = static_cast<std::uint8_t>(255 * (static_cast<float>(iteration)/max_iteration));
            // vertices[i].color = sf::Color {brightness, brightness, 150};
            vertices[i].color = sf::Color {iteration*200};
        }
    }

    while (main_window.isOpen()) {
        while (const std::optional event = main_window.pollEvent()) {
            if (event->is<sf::Event::Closed>())
                main_window.close();
        }

        main_window.clear();
        if (sf::Event::MouseWheelScrolled)
        main_window.draw(vertices);
        main_window.display();
    }
}