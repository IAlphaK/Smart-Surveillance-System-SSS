import express from 'express';
import { reportLabel } from '../controllers/labelControllers.js';

const router = express.Router();

router.post('/report/label', reportLabel);

export default router;
