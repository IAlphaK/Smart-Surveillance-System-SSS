import mongoose from 'mongoose';

const LoginSchema = new mongoose.Schema(
    {
        Username: {
            type: String,
            required: true 
        },
        password: {
            type: String,
            required: true
        }
    }
);

const Login = mongoose.model('Login', LoginSchema);
export default Login;