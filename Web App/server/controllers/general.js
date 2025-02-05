import Incident from "../models/Incident.js";
import Student from "../models/Student.js";

export const getSmokingIncidentCounts = async (req, res, internal = false) => {
    try {
        const smokingCounts = await Incident.aggregate([
            { $match: { behaviourDetected: "Smoking" } },
            {
                $group: {
                    _id: "$behaviorDetected",
                    count: { $sum: 1 }
                }
            }
        ]);

        if (internal) return smokingCounts;
        res.status(200).json(smokingCounts);
    } catch (error) {
        if (internal) throw error;
        res.status(500).json({ message: error.message });
    }
};



export const getFightingIncidentCounts = async (req, res, internal = false) => {
    try {
        const fightingCounts = await Incident.aggregate([
            { $match: { behaviourDetected: "Fighting" } },
            {
                $group: {
                    _id: "$behaviorDetected",
                    count: { $sum: 1 }
                }
            }
        ]);

        if (internal) return fightingCounts;
        res.status(200).json(fightingCounts);
    } catch (error) {
        if (internal) throw error;
        res.status(500).json({ message: error.message });
    }
};


export const getUniqueStudentCount = async (req, res, internal = false) => {
    try {
        const uniqueStudentCount = await Student.countDocuments(); // Counts all unique documents in the students collection

        if (internal) return uniqueStudentCount; // Return the count directly for internal calls
        res.status(200).json({ uniqueStudentCount }); // Send JSON response for external API calls
    } catch (error) {
        if (internal) throw error; // Throw the error to be handled by the calling function
        res.status(500).json({ message: error.message }); // Send error response for external API calls
    }
};

/*export const getAllIncidents = async (req, res, internal = false) => {
    try {
        const incidents = await Incident.find({});
        if (internal) return incidents;  // Return data for internal use
        // Setting headers to prevent caching
        res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
        res.setHeader("Pragma", "no-cache"); // For HTTP/1.0 compatibility
        res.setHeader("Expires", "0"); // Ensures proxies do not cache the response
        res.status(200).json(incidents);
    } catch (error) {
        if (internal) throw error;  // Throw to be caught by calling function
        res.status(500).json({ message: error.message });
    }
};*/

export const getAllIncidents = async (req, res, internal = false) => {
    try {
        // Use MongoDB's aggregate function to join incidents with students
        const incidents = await Incident.aggregate([
            {
                $lookup: {
                    from: "students",
                    localField: "studentID",
                    foreignField: "rollNumber",
                    as: "studentInfo"
                }
            },
            {
                $unwind: "$studentInfo"
            },
            {
                $project: {
                    cameraName: 1,
                    behaviourDetected: 1,
                    studentID: 1,
                    studentName: "$studentInfo.name",
                    time: 1, // Include the original time field
                    date: { $dateToString: { format: "%Y-%m-%d", date: "$time" } },  // Extract the date
                    timestamp: { $dateToString: { format: "%H:%M:%S", date: "$time" } }  // Extract the time
                }
            }
        ]);

        if (internal) return incidents;
        
        // Setting headers to prevent caching
        res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
        res.setHeader("Pragma", "no-cache");
        res.setHeader("Expires", "0");
        res.status(200).json(incidents);
    } catch (error) {
        if (internal) throw error;
        res.status(500).json({ message: error.message });
    }
};
