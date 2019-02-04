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
    
    // Loop through grid entries
    std::vector<std::vector<float>> allDifferences;
    for(const auto &g : grid) {
        const centimeter_t x = g.position[0];
        const centimeter_t y = g.position[1];
        
        // If snapshot is within R.O.I.
        if(x > 120_cm && x < 1200_cm && y > 0_cm && y < 1560_cm) {
            // Load snapshot and resize
            cv::Mat snapshot = g.loadGreyscale();
            cv::resize(snapshot, snapshot, imSize);

            // Get best heading using perfect memory
            degree_t bestHeading;
            size_t bestSnapshotIndex;
            float lowestDifference;
            std::tie(bestHeading, bestSnapshotIndex, lowestDifference, allDifferences) = pm.getHeading(snapshot);

            std::cout << "(" << x << ", " << y << ") : " << bestHeading << ", " << lowestDifference << ", " << bestSnapshotIndex;

            // Draw arrow showing vector field
            const centimeter_t xEnd = x + (60_cm * (1.0 - lowestDifference) * sin(-bestHeading));
            const centimeter_t yEnd = y + (60_cm * (1.0 - lowestDifference) * cos(-bestHeading));
            cv::arrowedLine(gridImage, cv::Point(x.value(), y.value()), cv::Point(xEnd.value(), yEnd.value()),
                            CV_RGB(0, 0, 255));

            // Get posiiton of best snapshot
            const centimeter_t bestSnapshotX = route[bestSnapshotIndex].position[0];
            const centimeter_t bestSnapshotY = route[bestSnapshotIndex].position[1];

            // If snapshot is less than 3m away i.e. algorithm hasn't entirely failed draw line from snapshot to route
            if(sqrt(((bestSnapshotX - x) * (bestSnapshotX - x)) + ((bestSnapshotY - y) * (bestSnapshotY - y))) < 3_m) {
                cv::line(gridImage, cv::Point(x.value(), y.value()), cv::Point(bestSnapshotX.value(), bestSnapshotY.value()),
                        CV_RGB(0, 255, 0));
                std::cout << " (good)" << std::endl;
            }
            std::cout << std::endl;
            cv::imwrite("grid_image.png", gridImage);
        }
    }
    
    
    return EXIT_SUCCESS;
}
