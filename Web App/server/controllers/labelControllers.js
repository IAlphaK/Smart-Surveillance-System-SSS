import { io } from '../index.js'; // Ensure this path matches the actual location of your index.js file
import Incident from '../models/Incident.js'; // Import the Incident model
import Student from '../models/Student.js'; // Import the Student model
import { fileURLToPath } from 'url';
import path from 'path';

// Get the __dirname equivalent in ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const reportLabel = (req, res) => {
    try {
        const { label, score, cameraName, personName } = req.body;
        console.log(`Label reported: ${label} (Score: ${score}) at Camera: ${cameraName} by Person: ${personName}`);
        // Check if an image was uploaded
        if (req.files && req.files.image) {
            const image = req.files.image;
            const uploadPath = path.join(__dirname, '../uploads/', image.name);

            // Save the image to the uploads folder
            image.mv(uploadPath, (err) => {
                if (err) {
                    console.error("Error saving image:", err);
                    return res.status(500).send({ error: "Failed to save image" });
                }

                console.log(`Image saved to ${uploadPath}`);

                // Emit an event to all connected clients, including the image path
                io.emit('labelAlert', {
                    label,
                    score,
                    cameraName,
                    personName: personName || "Unknown",
                    imagePath: req.files?.image ? `/uploads/${image.name}` : null
                });

                // Respond with success
                res.status(200).json({ message: `Label successfully reported: ${label}` });
            });
        } else {
            // Emit an event to all connected clients (without an image)
            //io.emit('labelAlert', { label, score, cameraName });
            io.emit('labelAlert', {
                label,
                score,
                cameraName,
                personName: personName || "Unknown", // Add the person's name if available, default to "Unknown"
                imagePath: req.files?.image ? `/uploads/${image.name}` : null
            });
            
            // Respond with success
            res.status(200).json({ message: `Label successfully reported: ${label}` });
        }
    } catch (error) {
        console.error("Error in reportLabel:", error);
        res.status(500).json({ error: "Internal Server Error" });
    }
};
