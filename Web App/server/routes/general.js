import express from "express";
import {
    getSmokingIncidentCounts,
    getFightingIncidentCounts,
    getUniqueStudentCount,
    getAllIncidents
} from "../controllers/general.js";
import Student from "../models/Student.js";

const router = express.Router();

router.get("/dashboard", async (req, res) => {
    try {
        const [smokingResult, fightingResult, uniqueStudentCount, allIncidents] = await Promise.all([
            getSmokingIncidentCounts(req, res, true),
            getFightingIncidentCounts(req, res, true),
            Student.countDocuments(),
            getAllIncidents(req, res, true)
        ]);

        const dashboardData = {
            smokingIncidentsCount: smokingResult.length > 0 ? smokingResult[0].count : 0,
            fightingIncidentsCount: fightingResult.length > 0 ? fightingResult[0].count : 0,
            uniqueStudentCount: uniqueStudentCount,
            incidents: allIncidents
        };

        res.status(200).json(dashboardData);
    } catch (error) {
        console.log(error);
        res.status(500).json({ message: error.message });
    }
});

export default router;