#pragma once

// BoB robotics 3rd party includes
#include "third_party/units.h"

// BoB robotics includes
#include "navigation/image_database.h"
#include "navigation/infomax.h"
#include "navigation/perfect_memory.h"
#include "navigation/perfect_memory_store_raw.h"

inline units::angle::degree_t shortestAngleBetween(units::angle::degree_t x, units::angle::degree_t y)
{
    return units::math::atan2(units::math::sin(x - y), units::math::cos(x - y));
}

//------------------------------------------------------------------------
// MemoryBase
//------------------------------------------------------------------------
class MemoryBase
{
public:
    MemoryBase()
    :   m_BestHeading(0), m_LowestDifference(std::numeric_limits<size_t>::max()), m_VectorLength(0)
    {
    }

    //------------------------------------------------------------------------
    // Declared virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t nearestRouteHeading) = 0;

    virtual void writeCSVHeader(std::ostream &os)
    {
        os << "Grid X [cm], Grid Y [cm], Best heading [degrees], Angular error [degrees], Lowest difference";
    }

    virtual void writeCSVLine(std::ostream &os, units::length::centimeter_t snapshotX, units::length::centimeter_t snapshotY, units::angle::degree_t angularError)
    {
        os << snapshotX << ", " << snapshotY << ", " << getBestHeading() << ", " << angularError << ", " << getLowestDifference();
    }

    virtual void render(cv::Mat &, units::length::centimeter_t, units::length::centimeter_t)
    {

    }

    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    units::angle::degree_t getBestHeading() const{ return m_BestHeading; }
    float getLowestDifference() const{ return m_LowestDifference; }
    float getVectorLength() const{ return m_VectorLength; }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    void setBestHeading(units::angle::degree_t bestHeading){ m_BestHeading = bestHeading; }
    void setLowestDifference(float lowestDifference){ m_LowestDifference = lowestDifference; }
    void setVectorLength(float vectorLength){ m_VectorLength = vectorLength; }

private:
    //-----------------------------------------------------------------------units-
    // Members
    //------------------------------------------------------------------------
    units::angle::degree_t m_BestHeading;
    float m_LowestDifference;
    float m_VectorLength;
};

//------------------------------------------------------------------------
// PerfectMemory
//------------------------------------------------------------------------
class PerfectMemory : public MemoryBase
{
public:
    PerfectMemory(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route,
                  bool renderGoodMatches, bool renderBadMatches)
    :   m_PM(imSize), m_Route(route), m_BestSnapshotIndex(std::numeric_limits<size_t>::max()),
        m_RenderGoodMatches(renderGoodMatches), m_RenderBadMatches(renderBadMatches)
    {
        m_PM.trainRoute(route, true);
        std::cout << "Trained on " << route.size() << " snapshots" << std::endl;
    }

    //------------------------------------------------------------------------
    // MemoryBase virtuals
    //------------------------------------------------------------------------
    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t) override
    {
        // Get heading directly from Perfect Memory
        units::angle::degree_t bestHeading;
        float lowestDifference;
        std::tie(bestHeading, m_BestSnapshotIndex, lowestDifference, std::ignore) = getPM().getHeading(snapshot);

        // Set best heading and vector length
        setBestHeading(snapshotHeading + bestHeading);
        setLowestDifference(lowestDifference);

        // Calculate vector length
        setVectorLength(1.0f - lowestDifference);
    }

    virtual void writeCSVHeader(std::ostream &os)
    {
        // Superclass
        MemoryBase::writeCSVHeader(os);

        os << ", Best snapshot index";
    }

    virtual void writeCSVLine(std::ostream &os, units::length::centimeter_t snapshotX, units::length::centimeter_t snapshotY, units::angle::degree_t angularError)
    {
        // Superclass
        MemoryBase::writeCSVLine(os, snapshotX, snapshotY, angularError);

        os << ", " << getBestSnapshotIndex();
    }

    virtual void render(cv::Mat &image, units::length::centimeter_t snapshotX, units::length::centimeter_t snapshotY)
    {
        // Get position of best snapshot
        const units::length::centimeter_t bestRouteX = m_Route[m_BestSnapshotIndex].position[0];
        const units::length::centimeter_t bestRouteY = m_Route[m_BestSnapshotIndex].position[1];

        // If snapshot is less than 3m away i.e. algorithm hasn't entirely failed draw line from snapshot to route
        const bool goodMatch = (units::math::sqrt(((bestRouteX - snapshotX) * (bestRouteX - snapshotX)) + ((bestRouteY - snapshotY) * (bestRouteY - snapshotY))) < units::length::meter_t(3));
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
    // Public API
    //------------------------------------------------------------------------
    size_t getBestSnapshotIndex() const{ return m_BestSnapshotIndex; }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const BoBRobotics::Navigation::PerfectMemoryRotater<> &getPM() const{ return m_PM; }

    void setBestSnapshotIndex(size_t bestSnapshotIndex){ m_BestSnapshotIndex = bestSnapshotIndex; }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    BoBRobotics::Navigation::PerfectMemoryRotater<> m_PM;
    const BoBRobotics::Navigation::ImageDatabase &m_Route;
    size_t m_BestSnapshotIndex;
    const bool m_RenderGoodMatches;
    const bool m_RenderBadMatches;
};

//------------------------------------------------------------------------
// PerfectMemoryConstrained
//------------------------------------------------------------------------
class PerfectMemoryConstrained : public PerfectMemory
{
public:
    PerfectMemoryConstrained(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route, units::angle::degree_t fov,
                             bool renderGoodMatches, bool renderBadMatches)
    :   PerfectMemory(imSize, route, renderGoodMatches, renderBadMatches), m_FOV(fov), m_ImageWidth(imSize.width)
    {
    }


    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t nearestRouteHeading) override
    {
        // Get 'matrix' of differences from perfect memory
        const auto &allDifferences = getPM().getImageDifferences(snapshot);

        // Loop through snapshots
        // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
        // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
        float lowestDifference = std::numeric_limits<float>::max();
        setBestSnapshotIndex(std::numeric_limits<size_t>::max());
        setBestHeading(units::angle::degree_t(0));
        for(size_t i = 0; i < allDifferences.size(); i++) {
            const auto &snapshotDifferences = allDifferences[i];

            // Loop through acceptable range of columns
            for(int c = 0; c < m_ImageWidth; c++) {
                // If this snapshot is a better match than current best
                if(snapshotDifferences[c] < lowestDifference) {
                    // Convert column into pixel rotation
                    int pixelRotation = c;
                    if(pixelRotation > (m_ImageWidth / 2)) {
                        pixelRotation -= m_ImageWidth;
                    }

                    // Convert this into angle
                    const units::angle::degree_t heading = snapshotHeading + units::angle::turn_t((double)pixelRotation / (double)m_ImageWidth);

                    // If the distance between this angle from grid and route angle is within FOV, update best
                    if(units::math::fabs(shortestAngleBetween(heading, nearestRouteHeading)) < m_FOV) {
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

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const units::angle::degree_t m_FOV;
    const int m_ImageWidth;
};

//------------------------------------------------------------------------
// InfoMax
//------------------------------------------------------------------------
class InfoMax : public MemoryBase
{
    using InfoMaxType = BoBRobotics::Navigation::InfoMaxRotater<BoBRobotics::Navigation::InSilicoRotater, float>;
    using InfoMaxWeightMatrixType = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

public:
    InfoMax(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route)
        : m_InfoMax(createInfoMax(imSize, route))
    {
    }

    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t) override
    {
        // Get heading directly from InfoMax
        units::angle::degree_t bestHeading;
        float lowestDifference;
        std::tie(bestHeading, lowestDifference, std::ignore) = m_InfoMax.getHeading(snapshot);

        // Set best heading and vector length
        setBestHeading(snapshotHeading + bestHeading);
        setLowestDifference(lowestDifference);

        // **TODO** calculate vector length
        setVectorLength(1.0f);
    }

protected:
    //------------------------------------------------------------------------
    // Protected API
    //------------------------------------------------------------------------
    const InfoMaxType &getInfoMax() const{ return m_InfoMax; }

private:
    //------------------------------------------------------------------------
    // Static API
    //------------------------------------------------------------------------
    // **TODO** move into BoB robotics
    static void writeWeights(const InfoMaxWeightMatrixType &weights, const filesystem::path &weightPath)
    {
        // Write weights to disk
        std::ofstream netFile(weightPath.str(), std::ios::binary);
        const int size[2] { (int) weights.rows(), (int) weights.cols() };
        netFile.write(reinterpret_cast<const char *>(size), sizeof(size));
        netFile.write(reinterpret_cast<const char *>(weights.data()), weights.size() * sizeof(float));
    }

    // **TODO** move into BoB robotics
    static InfoMaxWeightMatrixType readWeights(const filesystem::path &weightPath)
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

    static InfoMaxType createInfoMax(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route)
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
    // Members
    //------------------------------------------------------------------------
    InfoMaxType m_InfoMax;
};

//------------------------------------------------------------------------
// InfoMaxConstrained
//------------------------------------------------------------------------
class InfoMaxConstrained : public InfoMax
{
public:
    InfoMaxConstrained(const cv::Size &imSize, const BoBRobotics::Navigation::ImageDatabase &route, units::angle::degree_t fov)
        : InfoMax(imSize, route), m_FOV(fov), m_ImageWidth(imSize.width)
    {
    }

    virtual void test(const cv::Mat &snapshot, units::angle::degree_t snapshotHeading, units::angle::degree_t nearestRouteHeading) override
    {
        // Get vector of differences from InfoMax
        const auto &allDifferences = this->getInfoMax().getImageDifferences(snapshot);

        // Loop through snapshots
        // **NOTE** this currently uses a super-naive approach as more efficient solution is non-trivial because
        // columns that represent the rotations are not necessarily contiguous - there is a dis-continuity in the middle
        this->setLowestDifference(std::numeric_limits<float>::max());
        this->setBestHeading(units::angle::degree_t(0));
        for(size_t i = 0; i < allDifferences.size(); i++) {
            // If this snapshot is a better match than current best
            if(allDifferences[i] < this->getLowestDifference()) {
                // Convert column into pixel rotation
                int pixelRotation = i;
                if(pixelRotation > (m_ImageWidth / 2)) {
                    pixelRotation -= m_ImageWidth;
                }

                // Convert this into angle
                const units::angle::degree_t heading = snapshotHeading + units::angle::turn_t((double)pixelRotation / (double)m_ImageWidth);

                // If the distance between this angle from grid and route angle is within FOV, update best
                if(units::math::fabs(shortestAngleBetween(heading, nearestRouteHeading)) < m_FOV) {
                    this->setBestHeading(heading);
                    this->setLowestDifference(allDifferences[i]);
                }
            }
        }

        // **TODO** calculate vector length
        this->setVectorLength(1.0f);
    }

private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const units::angle::degree_t m_FOV;
    const int m_ImageWidth;
};
