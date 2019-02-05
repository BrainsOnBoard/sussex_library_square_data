// OpenCV
#include <opencv2/opencv.hpp>

// BoB robotics includes
#include "navigation/image_database.h"
#include "navigation/perfect_memory.h"
#include "navigation/perfect_memory_store_raw.h"

#include "psimpl.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::length;
using namespace units::angle;
using namespace units::math;

namespace
{
std::vector<cv::Point2i> decimateRoute(const Navigation::ImageDatabase &database)
{
    std::vector<float> routePointComponents;
    routePointComponents.reserve(database.size() * 2);
    for(const auto &r : database) {
        routePointComponents.emplace_back(centimeter_t(r.position[0]).value());
        routePointComponents.emplace_back(centimeter_t(r.position[1]).value());
    }

    std::vector<float> decimatedRoutePointComponents;
    psimpl::simplify_douglas_peucker<2>(routePointComponents.cbegin(), routePointComponents.cend(),
                                        15.0,
                                        std::back_inserter(decimatedRoutePointComponents));

    std::vector<cv::Point2i> decimatedRoutePoints;
    decimatedRoutePoints.reserve(decimatedRoutePointComponents.size() / 2);
    for(size_t i = 0; i < decimatedRoutePointComponents.size(); i += 2) {
        decimatedRoutePoints.emplace_back((int)std::round(decimatedRoutePointComponents[i]),
                                          (int)std::round(decimatedRoutePointComponents[i + 1]));
    }
    return decimatedRoutePoints;
}
}
int main()
{
    const cv::Size imSize(120, 25);

    // Default algorithm: find best-matching snapshot, use abs diff
    Navigation::PerfectMemoryRotater<> pm(imSize);
    
    // Train perfect memory, resizing images to fit
    Navigation::ImageDatabase route("routes/route5/skymask");
    pm.trainRoute(route, true);
    std::cout << "Trained on " << pm.getNumSnapshots() << " snapshots" << std::endl;
    
    const auto decimatedRoutePoints = decimateRoute(route);
    cv::Mat decimatedRoutePointMat(decimatedRoutePoints, true);

    // Load grid
    Navigation::ImageDatabase grid("image_grids/mid_day/skymask");
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
    cv::polylines(gridImage, routePointsMat, false, CV_RGB(128, 128, 128));
    cv::polylines(gridImage, decimatedRoutePointMat, false, CV_RGB(255, 255, 255));
    bool showBadMatches = false;
    
    // Loop through grid entries
    std::vector<std::vector<float>> allDifferences;
    for(const auto &g : grid) {
        const centimeter_t x = g.position[0];
        const centimeter_t y = g.position[1];
        
        // If snapshot is within R.O.I.
        if(x > 240_cm && x < 1200_cm && y > 0_cm && y < 1560_cm) {
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
            const bool goodMatch = (sqrt(((bestSnapshotX - x) * (bestSnapshotX - x)) + ((bestSnapshotY - y) * (bestSnapshotY - y))) < 3_m);
            if(goodMatch || showBadMatches) {
                cv::line(gridImage, cv::Point(x.value(), y.value()), cv::Point(bestSnapshotX.value(), bestSnapshotY.value()),
                        goodMatch ? CV_RGB(0, 255, 0) : CV_RGB(255, 0, 0));
            }
            
            if(goodMatch) {
                std::cout << " (good)";
            }
            std::cout << std::endl;
            
            /*minDifferences.clear();
            for(size_t i = 0; i < pm.getNumSnapshots(); i++) {
                const auto elem = std::min_element(std::cbegin(allDifferences[i]), std::cend(allDifferences[i]));
                minDifferences.emplace(*elem, i);
            }
            
            size_t i = 0;
            auto d = minDifferences.cbegin();
            for(;i < 100; i++, d++) {
                std::cout << d->second << ":" << d->first << std::endl;
            }*/
            cv::imwrite("grid_image.png", gridImage);
        }
    }
    
    
    return EXIT_SUCCESS;
}
