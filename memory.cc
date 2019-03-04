#include "memory.h"

using namespace BoBRobotics;
using namespace units::literals;
using namespace units::length;
using namespace units::angle;
using namespace units::math;

//------------------------------------------------------------------------
// MemoryBase
//------------------------------------------------------------------------
MemoryBase::MemoryBase(const cv::Size &imSize)
:   m_BestHeading(0), m_LowestDifference(std::numeric_limits<size_t>::max()), m_VectorLength(0), m_ImageSize(imSize)
{
}
//------------------------------------------------------------------------
void MemoryBase::writeCSVHeader(std::ostream &os)
{
    os << "Grid X [cm], Grid Y [cm], Best heading [degrees], Angular error [degrees], Lowest difference";
}
//------------------------------------------------------------------------
void MemoryBase::writeCSVLine(std::ostream &os, centimeter_t snapshotX, centimeter_t snapshotY, degree_t angularError)
{
    os << snapshotX << ", " << snapshotY << ", " << getBestHeading() << ", " << angularError << ", " << getLowestDifference();
}


//------------------------------------------------------------------------
// PerfectMemory
//------------------------------------------------------------------------
PerfectMemory::PerfectMemory(const cv::Size &imSize, const Navigation::ImageDatabase &route,
                             bool renderGoodMatches, bool renderBadMatches)
:   MemoryBase(imSize), m_PM(imSize), m_Route(route), m_BestSnapshotIndex(std::numeric_limits<size_t>::max()),
    m_RenderGoodMatches(renderGoodMatches), m_RenderBadMatches(renderBadMatches)
{
    m_PM.trainRoute(route, true);
    std::cout << "Trained on " << route.size() << " snapshots" << std::endl;
}
//------------------------------------------------------------------------
void PerfectMemory::test(const cv::Mat &snapshot, degree_t snapshotHeading, degree_t)
{
    // Get heading directly from Perfect Memory
    degree_t bestHeading;
    float lowestDifference;
    std::tie(bestHeading, m_BestSnapshotIndex, lowestDifference, std::ignore) = getPM().getHeading(snapshot);

    // Set best heading and vector length
    setBestHeading(snapshotHeading + bestHeading);
    setLowestDifference(lowestDifference);

    // Calculate vector length
    setVectorLength(1.0f - lowestDifference);
}
//------------------------------------------------------------------------
std::vector<float> PerfectMemory::calculateRIDF(const cv::Mat &snapshot) const
{
    // Get 'matrix' of differences from perfect memory
    const auto &allDifferences = getPM().getImageDifferences(snapshot);

    // Reserve vector to hold RIDF
    std::vector<float> ridf(getImageSize().width, std::numeric_limits<float>::max());

    // Loop through all snapshots
    for(const auto &memDifferences : allDifferences) {
        for(int c = 0; c < getImageSize().width; c++) {
            ridf[c] = std::min(ridf[c], memDifferences[c]);
        }
    }

    return ridf;
}
//------------------------------------------------------------------------
void PerfectMemory::writeCSVHeader(std::ostream &os)
{
    // Superclass
    MemoryBase::writeCSVHeader(os);

    os << ", Best snapshot index";
}
//------------------------------------------------------------------------
void PerfectMemory::writeCSVLine(std::ostream &os, centimeter_t snapshotX, centimeter_t snapshotY, degree_t angularError)
{
    // Superclass
    MemoryBase::writeCSVLine(os, snapshotX, snapshotY, angularError);

    os << ", " << getBestSnapshotIndex();
}
//------------------------------------------------------------------------
void PerfectMemory::render(cv::Mat &image, centimeter_t snapshotX, centimeter_t snapshotY)
{
    // Get position of best snapshot
    const centimeter_t bestRouteX = m_Route[m_BestSnapshotIndex].position[0];
    const centimeter_t bestRouteY = m_Route[m_BestSnapshotIndex].position[1];

    // If snapshot is less than 3m away i.e. algorithm hasn't entirely failed draw line from snapshot to route
    const bool goodMatch = (sqrt(((bestRouteX - snapshotX) * (bestRouteX - snapshotX)) + ((bestRouteY - snapshotY) * (bestRouteY - snapshotY))) < 3_m);
    if(goodMatch && m_RenderGoodMatches) {
        cv::line(image, cv::Point(snapshotX.value(), snapshotY.value()), cv::Point(bestRouteX.value(), bestRouteY.value()),
                    CV_RGB(0, 255, 0));
    }
    else if(!goodMatch && m_RenderBadMatches) {
        cv::line(image, cv::Point(snapshotX.value(), snapshotY.value()), cv::Point(bestRouteX.value(), bestRouteY.value()),
                    CV_RGB(255, 0, 0));
    }
}



//------------------------------------------------------------------------
// PerfectMemoryConstrained
//------------------------------------------------------------------------
PerfectMemoryConstrained::PerfectMemoryConstrained(const cv::Size &imSize, const Navigation::ImageDatabase &route, degree_t fov,
                                                   bool renderGoodMatches, bool renderBadMatches)
:   PerfectMemory(imSize, route, renderGoodMatches, renderBadMatches), m_FOV(fov)
{
}
//------------------------------------------------------------------------
void PerfectMemoryConstrained::test(const cv::Mat &snapshot, degree_t snapshotHeading, degree_t nearestRouteHeading)
{
    // Get 'matrix' of differences from perfect memory
    const auto &allDifferences = getPM().getImageDifferences(snapshot);

    // Loop through snapshots
    // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
    // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
    float lowestDifference = std::numeric_limits<float>::max();
    setBestSnapshotIndex(std::numeric_limits<size_t>::max());
    setBestHeading(0_deg);
    for(size_t i = 0; i < allDifferences.size(); i++) {
        const auto &snapshotDifferences = allDifferences[i];

        // Loop through acceptable ram_ImageWidthnge of columns
        for(int c = 0; c < getImageSize().width; c++) {
            // If this snapshot is a better match than current best
            if(snapshotDifferences[c] < lowestDifference) {
                // Convert column into pixel rotation
                int pixelRotation = c;
                if(pixelRotation > (getImageSize().width / 2)) {
                    pixelRotation -= getImageSize().width;
                }

                // Convert this into angle
                const degree_t heading = snapshotHeading + turn_t((double)pixelRotation / (double)getImageSize().width);

                // If the distance between this angle from grid and route angle is within FOV, update best
                if(fabs(shortestAngleBetween(heading, nearestRouteHeading)) < m_FOV) {
                    setBestSnapshotIndex(i);
                    setBestHeading(heading);
                    lowestDifference = snapshotDifferences[c];
                }
            }
        }
    }

    // Check valid snapshot actually exists
    assert(getBestSnapshotIndex() != std::numeric_limits<size_t>::max());

    // Scale difference to match code in ridf_processors.h:57
    setLowestDifference(lowestDifference / 255.0f);

    // Calculate vector length
    setVectorLength(1.0f - getLowestDifference());
}

//------------------------------------------------------------------------
// InfoMax
//------------------------------------------------------------------------
InfoMax::InfoMax(const cv::Size &imSize, const Navigation::ImageDatabase &route)
    : MemoryBase(imSize), m_InfoMax(createInfoMax(imSize, route))
{
}
//------------------------------------------------------------------------
void InfoMax::test(const cv::Mat &snapshot, degree_t snapshotHeading, degree_t)
{
    // Get heading directly from InfoMax
    degree_t bestHeading;
    float lowestDifference;
    std::tie(bestHeading, lowestDifference, std::ignore) = m_InfoMax.getHeading(snapshot);

    // Set best heading and vector length
    setBestHeading(snapshotHeading + bestHeading);
    setLowestDifference(lowestDifference);

    // **TODO** calculate vector length
    setVectorLength(1.0f);
}
//------------------------------------------------------------------------
std::vector<float> InfoMax::calculateRIDF(const cv::Mat &snapshot) const
{
    return getInfoMax().getImageDifferences(snapshot);
}
//------------------------------------------------------------------------
void InfoMax::writeWeights(const InfoMax::InfoMaxWeightMatrixType &weights, const filesystem::path &weightPath)
{
    // Write weights to disk
    std::ofstream netFile(weightPath.str(), std::ios::binary);
    const int size[2] { (int) weights.rows(), (int) weights.cols() };
    netFile.write(reinterpret_cast<const char *>(size), sizeof(size));
    netFile.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(float));
}
//------------------------------------------------------------------------
// **TODO** move into BoB robotics
InfoMax::InfoMaxWeightMatrixType InfoMax::readWeights(const filesystem::path &weightPath)
{
    // Open file
    std::ifstream is(weightPath.str(), std::ios::binary);
    if (!is.good()) {
        throw std::runtime_error("Could not open " + weightPath.str());
    }

    // The matrix size is encoded as 2 x int32_t
    int32_t size[2];
    is.read(reinterpret_cast<char *>(&size), sizeof(size));

    // Create data array and fill it
    InfoMaxWeightMatrixType data(size[0], size[1]);
    is.read(reinterpret_cast<char *>(data.data()), sizeof(float) * data.size());

    return std::move(data);
}
//------------------------------------------------------------------------
InfoMax::InfoMaxType InfoMax::createInfoMax(const cv::Size &imSize, const Navigation::ImageDatabase &route)
{
    // Create path to weights from directory containing route
    const filesystem::path weightPath = filesystem::path(route.getPath()) / "infomax.bin";
    if(weightPath.exists()) {
        std::cout << "Loading weights from " << weightPath << std::endl;
        InfoMaxType infomax(imSize, readWeights(weightPath));
        return std::move(infomax);
    }
    else {
        InfoMaxType infomax(imSize);
        infomax.trainRoute(route, true);
        writeWeights(infomax.getWeights(), weightPath.str());
        std::cout << "Trained on " << route.size() << " snapshots" << std::endl;
        return std::move(infomax);
    }
}


//------------------------------------------------------------------------
// InfoMaxConstrained
//------------------------------------------------------------------------
InfoMaxConstrained::InfoMaxConstrained(const cv::Size &imSize, const Navigation::ImageDatabase &route, degree_t fov)
:   InfoMax(imSize, route), m_FOV(fov)
{
}
//------------------------------------------------------------------------
void InfoMaxConstrained::test(const cv::Mat &snapshot, degree_t snapshotHeading, degree_t nearestRouteHeading)
{
    // Get vector of differences from InfoMax
    const auto &allDifferences = getInfoMax().getImageDifferences(snapshot);

    // Loop through snapshots
    // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
    // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
    setLowestDifference(std::numeric_limits<float>::max());
    setBestHeading(0_deg);
    for(size_t i = 0; i < allDifferences.size(); i++) {
        // If this snapshot is a better match than current best
        if(allDifferences[i] < getLowestDifference()) {
            // Convert column into pixel rotation
            int pixelRotation = i;
            if(pixelRotation > (getImageSize().width / 2)) {
                pixelRotation -= getImageSize().width;
            }

            // Convert this into angle
            const degree_t heading = snapshotHeading + turn_t((double)pixelRotation / (double)getImageSize().width);

            // If the distance between this angle from grid and route angle is within FOV, update best
            if(fabs(shortestAngleBetween(heading, nearestRouteHeading)) < m_FOV) {
                setBestHeading(heading);
                setLowestDifference(allDifferences[i]);
            }
        }
    }

    // **TODO** calculate vector length
    setVectorLength(1.0f);
}
