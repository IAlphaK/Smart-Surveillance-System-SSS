import { io } from '../index.js'; // Ensure this path matches the actual location of your index.js file

export const reportIncident = (req, res) => {
    const { rollNumber, cameraName } = req.body;
    console.log(`Incident reported for Roll Number: ${rollNumber} at Camera: ${cameraName}`);

    // Emit an event to all connected clients
    io.emit('alert', { rollNumber, cameraName });

    res.status(200).json({ message: `Incident successfully reported for ${rollNumber}.` });
};
