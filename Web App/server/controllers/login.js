// controllers/login.js
import Login from "../models/Login.js";
import bcrypt from 'bcrypt';

export const loginUser = async (req, res) => {
    try {
        const { Username, password } = req.body;
        const user = await Login.findOne({ Username });

        // Change here currently it is browser alert. It is supposed to be login alert on interface based on invalid user/password.
        if (!user) {
            return res.status(404).json({ message: "User not found" });
        }

        console.log(user.password);
        console.log(password);
        // Check if the provided password matches the stored password
        const isMatch = await bcrypt.compare(password, user.password);
       
        // Change here currently it is browser alert. It is supposed to be login alert on interface based on invalid user/password.
        if (!isMatch) {
            return res.status(400).json({ message: "Invalid credentials" });
        }

        // Assuming a function to create a token exists
        const token = createToken(user._id);

        // Setting headers to prevent caching
        res.setHeader("Cache-Control", "no-cache, no-store, must-revalidate");
        res.setHeader("Pragma", "no-cache"); // For HTTP/1.0 compatibility
        res.setHeader("Expires", "0"); // Ensures proxies do not cache the response

        res.status(200).json({ user: user.Username, token });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

// Function to create a token (pseudo code)
const createToken = (userId) => {
    // Implement token creation logic here, possibly using JWT
    return "some_generated_token";
};
