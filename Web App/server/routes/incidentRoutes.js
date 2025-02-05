import express from 'express';
import { reportIncident } from '../controllers/incidentController.js';

const router = express.Router();

router.post('/report/incident', reportIncident);

export default router;
