import express from 'express';
import bodyParser from 'body-parser';
import mongoose from 'mongoose';
import cors from 'cors';
import dotenv from 'dotenv';
import helmet from 'helmet';
import morgan from 'morgan';
import http from 'http';
import { Server } from 'socket.io';
import fileUpload from 'express-fileupload';

// Route imports
import generalRoutes from "./routes/general.js";
import summaryRoutes from "./routes/summary.js";
import reportsRoutes from "./routes/reports.js";
import incidentRoutes from './routes/incidentRoutes.js';

import loginRoutes from "./routes/login.js"; // Import the login routes
import labelRoutes from './routes/labelRoutes.js';


// Model imports
import Student from "./models/Student.js";
import {dummyStudents} from "./data/index.js";
import Incident from "./models/Incident.js";
import { dummyIncidents } from './data/index.js';
import Login from "./models/Login.js";
import { logindata} from "./data/index.js";

// Project Configurations
dotenv.config();
const app = express();

const server = http.createServer(app);
export const io = new Server(server, {  // Export the 'io' instance
    cors: {
        origin: ["http://localhost:3000"],
        methods: ["GET", "POST"],
        credentials: true
    }
});

// Middleware
app.use(express.json());
app.use(helmet());
app.use(helmet.crossOriginResourcePolicy({policy: "cross-origin"}));
app.use(morgan("common"));
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({extended: true}));
app.use(cors());

// Add fileUpload middleware
app.use(fileUpload());
// Serve static files from the 'uploads' directory
app.use('/uploads', express.static('uploads'));

// Routes with Socket.IO passed as an argument
app.use(incidentRoutes);
app.use("/general", generalRoutes);
app.use("/summary", summaryRoutes);
app.use("/reports", reportsRoutes);

app.use("/auth", loginRoutes); // Use the login routes
// Add the new route for label reporting
app.use(labelRoutes);

// Helper function to capitalize the first letter
const capitalizeLabel = (label) => {
    if (!label) return label;
    return label.charAt(0).toUpperCase() + label.slice(1);
};

// Socket.IO acknowledgment listener
io.on('connection', (socket) => {
// In server's index.js, update the socket acknowledgment handler:
    socket.on('acknowledge', async (data) => {
        let { label, cameraName, personName } = data; 
        
        // Capitalize the first letter of the label
        label = capitalizeLabel(label);

        try {
            // Find the corresponding student name from dummyStudents
            const studentRecord = dummyStudents.find(student => student.rollNumber === personName);
            const studentName = studentRecord ? studentRecord.name : "Unknown";

            // Create new incident record
            const newIncident = new Incident({
                cameraName: cameraName,
                behaviourDetected: label,
                studentID: personName || "Unknown", // This will be the roll number
                time: new Date(),
            });

            await newIncident.save();

            // Emit an event to all clients that a new incident has been added
            io.emit('newIncident', { message: 'New incident added' });

            // Check if a student with this rollNumber already exists in the database
            const existingStudent = await Student.findOne({ rollNumber: personName });

            // If not, create a new student record with both rollNumber and name
            if (!existingStudent && personName && personName !== "Unknown") {
                const newStudent = new Student({
                    rollNumber: personName,
                    name: studentName, // Use the mapped name from dummyStudents
                });

                await newStudent.save();
                console.log('New student record saved to database.');
            }

            console.log('Incident data saved to database with student:', {
                rollNumber: personName,
                name: studentName
            });
        } catch (error) {
            console.error('Error saving data to database:', error);
        }
    });
});



// MongoDB connection and server start
const PORT = process.env.PORT || 9000;
mongoose.connect(process.env.MONGO_URL, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
})
.then(() => {
    server.listen(PORT, () => console.log(`Server running on port: ${PORT}`));


    /*Student.insertMany(dummyStudents)
            .then(() => console.log("Students data inserted"))
            .catch((error) => console.log("Error inserting students data:", error));

    Incident.insertMany(dummyIncidents)
            .then(() => console.log("Incidents data inserted"))
            .catch((error) => console.log("Error inserting incidents data:", error));
    /*Login.insertMany(logindata);*/
})
.catch(error => console.log(`${error} did not connect`));
