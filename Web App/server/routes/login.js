// routes/login.js
import express from "express";
import { loginUser } from "../controllers/login.js";

const router = express.Router();

// Route to handle POST request for user login
router.post("/login", loginUser);

export default router;