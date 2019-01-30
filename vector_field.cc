// OpenCV
#include <opencv2/opencv.hpp>

// BoB robotics includes
#include "navigation/image_database.h"
#include "navigation/perfect_memory.h"
#include "navigation/perfect_memory_store_raw.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::length;
using namespace units::angle;
using namespace units::math;

int main()
{
    const cv::Size imSize(120, 25);

    // Default algorithm: find best-matching snapshot, use abs diff
    Navigation::PerfectMemoryRotater<> pm(imSize);
    
    // Train perfect memory, resizing images to fit
    Navigation::ImageDatabase route("processed_routes/route5/");
    pm.trainRoute(route, true);
    std::cout << "Trained on " << pm.getNumSnapshots() << " snapshots" << std::endl;
    
    // Load grid
    Navigation::ImageDatabase grid("unwrapped_image_grid/mid_day/");
    assert(grid.isGrid());
    assert(grid.hasMetadata());
    
    // Read grid dimensions from meta data
    std::vector<double> size, seperationMM;
    grid.getMetadata()["grid"]["separationMM"] >> seperationMM;
    grid.getMetadata()["grid"]["size"] >> size;
    assert(size.size() == 3);
    assert(seperationMM.size() == 3);
    
    std::cout << size[0] << "x" << size[1] << " grid with " << seperationMM[0] << "x" << seperationMM[1] << "mm squares" << std::endl;
    
    // Make a grid image with one pixel per cm
    cv::Mat gridImage((int)std::round(size[1] * seperationMM[1] * 0.1), 
                      (int)std::round(size[0] * seperationMM[0] * 0.1), 
                      CV_8UC3, cv::Scalar::all(0));
    
    // **TODO** could probably work directly into correctly formatted cv::Mat
    std::vector<cv::Point2i> routePoints;
    routePoints.reserve(route.size());
    for(const auto &r : route) {
        const centimeter_t x = r.position[0];
        const centimeter_t y = r.position[1];
        
        routePoints.emplace_back((int)std::round(x.value()), (int)std::round(y.value()));
    }
    
    // Draw route onto image
    cv::Mat routePointsMat(routePoints, true);
    cv::polylines(gridImage, routePointsMat, false, CV_RGB(255, 0, 0));
    
    /*degree_t lastHeading = 0_deg;
    for(const auto &r : route) {
        const degree_t heading = r.heading;
        
        if(heading != lastHeading) {
            const centimeter_t x = r.position[0];
            const centimeter_t y = r.position[1];
            
            const centimeter_t xEnd = x + (120_cm * cos(heading));
            const centimeter_t yEnd = y + (120_cm * sin(heading));
            cv::arrowedLine(gridImage, cv::Point(x.value(), y.value()), cv::Point(xEnd.value(), yEnd.value()),
                            CV_RGB(0, 0, 255));
            lastHeading = heading;
        }
    }*/
    
    // Loop through grid entries
    std::vector<std::vector<float>> allDifferences;
    for(const auto &g : grid) {
        const centimeter_t x = g.position[0];
        const centimeter_t y = g.position[1];
        
        // Load snapshot and resize
        cv::Mat snapshot = g.loadGreyscale();
        cv::resize(snapshot, snapshot, imSize);
        
        degree_t bestHeadingRelative;
        size_t bestSnapshotIndex;
        float lowestDifference;
        std::tie(bestHeadingRelative, bestSnapshotIndex, lowestDifference, allDifferences) = pm.getHeading(snapshot);
        
        const degree_t bestSnapshotHeading = route[bestSnapshotIndex].heading;
        std::cout << "(" << x << ", " << y << ") : " << bestHeadingRelative << ", " << bestSnapshotHeading << ", " << lowestDifference << ", " << bestSnapshotIndex << std::endl;
        
        const centimeter_t xEnd = x + (60_cm * (1.0 - lowestDifference) * cos(bestSnapshotHeading - bestHeadingRelative));
        const centimeter_t yEnd = y + (60_cm * (1.0 - lowestDifference) * sin(bestSnapshotHeading - bestHeadingRelative));
        cv::arrowedLine(gridImage, cv::Point(x.value(), y.value()), cv::Point(xEnd.value(), yEnd.value()),
                        CV_RGB(0, 0, 255));
        cv::imwrite("grid_image.png", gridImage);
    }
    
    
    return EXIT_SUCCESS;
}
