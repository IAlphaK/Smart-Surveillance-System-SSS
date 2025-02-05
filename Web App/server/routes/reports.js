import express from 'express';
import { generateReport } from '../controllers/reports.js';

const router = express.Router();

router.get('/download', generateReport);

export default router;
