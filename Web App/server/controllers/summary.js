import Incident from "../models/Incident.js";
import Student from "../models/Student.js";

/*export const getIncidentDetails = async (req, res) => {
  try {
    const incidents = await Incident.find().populate("studentID");
    res.status(200).json(incidents);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};*/

export const getStudentDetails = async (req, res) => {
    try {
      const { id } = req.params;
      const student = await Student.findById(id);
  
      // Setting headers to prevent caching
      res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
      res.setHeader("Pragma", "no-cache"); // For HTTP/1.0 compatibility
      res.setHeader("Expires", "0"); // Ensures proxies do not cache the response
      console.log(student);
      res.status(200).json(student);
    } catch (error) {
      res.status(500).json({ message: error.message });
    }
};  

/*export const getSmokers = async(req, res) => {
    try {
        const smokers = await Incident.find({behaviourDetected: "Smoking"});
    
        // Setting headers to prevent caching
        res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
        res.setHeader("Pragma", "no-cache"); // For HTTP/1.0 compatibility
        res.setHeader("Expires", "0"); // Ensures proxies do not cache the response
        
        res.status(200).json(smokers);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};*/

export const getSmokers = async (req, res) => {
    try {
        // Use aggregate to join with students collection
        const smokers = await Incident.aggregate([
            { $match: { behaviourDetected: "Smoking" } },
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
                    date: { $dateToString: { format: "%Y-%m-%d", date: "$time" } },  // Extract date
                    timestamp: { $dateToString: { format: "%H:%M:%S", date: "$time" } }  // Extract time
                }
            }
        ]);
  
        // Setting headers to prevent caching
        res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
        res.setHeader("Pragma", "no-cache");
        res.setHeader("Expires", "0");
  
        res.status(200).json(smokers);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
  };


/*export const getFighters = async(req, res) => {
    try{
        const fighter = await Incident.find({behaviourDetected: "Fighting"});
    
        // Setting headers to prevent caching
        res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
        res.setHeader("Pragma", "no-cache"); // For HTTP/1.0 compatibility
        res.setHeader("Expires", "0"); // Ensures proxies do not cache the response
        
        res.status(200).json(fighter);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
}*/

export const getFighters = async (req, res) => {
    try {
        // Use aggregate to join with students collection
        const fighters = await Incident.aggregate([
            { $match: { behaviourDetected: "Fighting" } },
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
                    date: { $dateToString: { format: "%Y-%m-%d", date: "$time" } },  // Extract date
                    timestamp: { $dateToString: { format: "%H:%M:%S", date: "$time" } }  // Extract time
                }
            }
        ]);
  
        // Setting headers to prevent caching
        res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
        res.setHeader("Pragma", "no-cache");
        res.setHeader("Expires", "0");
  
        res.status(200).json(fighters);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
  };
