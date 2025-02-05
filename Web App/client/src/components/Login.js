import React, { useState, useEffect } from 'react';
import { TextField, Button, Box, Typography, Alert } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import { useDispatch } from 'react-redux';
import { setAuthStatus } from '../state/index';
import axios from 'axios';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [errorMessage, setErrorMessage] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [videoOpacity, setVideoOpacity] = useState(1);
  const navigate = useNavigate();
  const dispatch = useDispatch();

  useEffect(() => {
    // Show form after 3 seconds
    const timer = setTimeout(() => {
      setShowForm(true);
      // Fade the background video
      setVideoOpacity(0.4); // Adjust opacity for the fade effect
    }, 3000);

    // Clear timer on component unmount
    return () => clearTimeout(timer);
  }, []);

  const handleLogin = async (event) => {
    event.preventDefault();
    try {
      const response = await axios.post(`${process.env.REACT_APP_BASE_URL}/auth/login`, {
        Username: username,
        password: password
      });

      localStorage.setItem('token', response.data.token);
      dispatch(setAuthStatus({ isAuthenticated: true, token: response.data.token }));
      navigate('/dashboard');
    } catch (error) {
      if(error.response && error.response.data) {
        setErrorMessage(error.response.data.message);
      }
      else{
        setErrorMessage('Login Failed. Please try again.');
      }
    }
  };

  return (
    <Box
      sx={{
        position: 'relative',
        width: '100%',
        height: '100vh',
        overflow: 'hidden',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      {/* Background Video */}
      <video
        autoPlay
        loop
        muted
        style={{
          position: 'absolute',
          width: '100%',
          height: '100%',
          objectFit: 'cover',
          zIndex: -1,
          opacity: videoOpacity,
          transition: 'opacity 1.5s ease-in-out', // Smooth transition for video
        }}
      >
        <source src={require('../Glitch Video - Made with Clipchamp.mp4')} type="video/mp4" />
      </video>

      {/* Dark Overlay */}
      {showForm && (
        <Box
          sx={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            backgroundColor: 'rgba(0, 0, 0, 0.6)', // Black shade overlay
            zIndex: -0.5, // Layer between video and form
            transition: 'background-color 1.5s ease-in-out', // Smooth transition for the black shade
          }}
        />
      )}

      {/* Login Form */}
      {showForm && (
        <Box
          sx={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            opacity: showForm ? 1 : 0, // Control opacity for smooth fade-in
            transition: 'opacity 2s ease-in-out', // Smoother and slower transition for form
            transform: showForm ? 'scale(1)' : 'scale(0.9)', // Slight zoom-in effect
            transitionProperty: 'opacity, transform',
            transitionDuration: '2s', // Matching duration for opacity and transform
            background: 'rgba(255, 255, 255, 0.1)', // Optional: slight background to form
            padding: 4,
            borderRadius: 2,
            zIndex: 1, // Ensure the form is above the overlay
          }}
        >
          <Typography component="h1" variant="h5">
            SMART SURVEILLANCE SYSTEM
          </Typography>
          <Box component="form" onSubmit={handleLogin} sx={{ mt: 1 }}>
            <TextField
              margin="normal"
              required
              fullWidth
              id="username"
              label="Username"
              name="username"
              autoComplete="username"
              autoFocus
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              sx={{
                transition: 'opacity 2s ease-in-out', // Smooth fade for input fields
              }}
            />
            <TextField
              margin="normal"
              required
              fullWidth
              name="password"
              label="Password"
              type="password"
              id="password"
              autoComplete="current-password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              sx={{
                transition: 'opacity 2s ease-in-out',
              }}
            />

            {errorMessage && (
              <Alert severity="error" sx={{ mt: 2, width: '100%' }}>
                {errorMessage}
              </Alert>
            )}

            <Button
              type="submit"
              fullWidth
              variant="contained"
              sx={{ mt: 3, mb: 2, transition: 'opacity 2s ease-in-out' }} // Smooth fade for button
            >
              Sign In
            </Button>
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default Login;
