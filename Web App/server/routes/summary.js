import express from "express";
import { /*getIncidentDetails,*/ getStudentDetails, getSmokers, getFighters } from "../controllers/summary.js";

const router = express.Router();
// end points are defined here which you will use in the API call in the frontend folder state/api.js 
/*router.get("/incidents", getIncidentDetails);*/
router.get("/student/:id", getStudentDetails);
router.get("/smokers", getSmokers);
router.get("/fighters", getFighters);
export default router;