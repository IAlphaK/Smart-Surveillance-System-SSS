import mongoose from 'mongoose';

const IncidentSchema = new mongoose.Schema(
    {
        time: {
            type: Date,
            required: true       
        }, 
        cameraName: {
            type: String,
            required: true
        },
        behaviourDetected: {
            type: String,
            required: true,
        },
        studentID: {
            type: String,
            ref: 'Student',
            required: true
        }
    }
);

const Incident = mongoose.model('Incident', IncidentSchema);
export default Incident;