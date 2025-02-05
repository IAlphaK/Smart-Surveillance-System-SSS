import React, { useMemo, useEffect } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { BrowserRouter, Route, Routes, Navigate } from 'react-router-dom';
import { CssBaseline, ThemeProvider } from '@mui/material';
import { createTheme } from '@mui/material/styles';
import io from "socket.io-client";
import Swal from 'sweetalert2';
import axios from 'axios';
import Dashboard from "scenes/dashboard";
import Layout from "scenes/layout";
import Smokers from "scenes/smokers";
import Fighters from "scenes/fighters";
import Login from "scenes/login";
import { themeSettings } from "theme";
import { setAuthStatus } from "./state/index";

function App() {
  const dispatch = useDispatch();
  const mode = useSelector((state) => state.global.mode);
  const theme = useMemo(() => createTheme(themeSettings(mode)), [mode]);

  // Check if token exists in localStorage and set authentication state accordingly
  useEffect(() => {
    const token = localStorage.getItem('token');
    if (token) {
      dispatch(setAuthStatus({ isAuthenticated: true, token }));
    }
  }, [dispatch]);

  const isAuthenticated = useSelector((state) => state.global.isAuthenticated);

  const handleLogout = () => {
    // Clear token from local storage
    localStorage.removeItem('token');
    // Update state to reflect user is not authenticated
    dispatch(setAuthStatus({ isAuthenticated: false, token: null }));
    // Redirect to login
    window.location.href = '/login'; // Use `window.location.href` for a complete refresh to login
  };

  useEffect(() => {
    const baseURL = process.env.REACT_APP_BASE_URL;
    const socket = io(`${baseURL}`);

    

    // New label alert listener for kinectics_inference.py
    socket.on('labelAlert', (alert) => {
      Swal.fire({
        title: 'Action Detected',
        text: `ALERT: ${alert.personName} caught ${alert.label} at ${alert.cameraName}`,
        icon: 'info',
        showCancelButton: true,
        confirmButtonText: 'Acknowledged',
        showDenyButton: !!alert.imagePath,  // Show View Image button if imagePath exists
        denyButtonText: 'View Image',
        allowOutsideClick: false, // Prevent clicking outside the alert to close it
        allowEscapeKey: false,   // Prevent the escape key from closing the alert
        preDeny: () => {
          // Open the image in a new window when 'View Image' is clicked and prevent modal from closing
          window.open(`${baseURL}${alert.imagePath}`, '_blank');
          return false;  // Returning false prevents the Swal from closing
        }
      }).then((result) => {
        if (result.isConfirmed) {
          // Emit acknowledgment event to the server, passing label and cameraName
          socket.emit('acknowledge', {
            label: alert.label,
            cameraName: alert.cameraName,
            personName: alert.personName
          });
          console.log('Acknowledgment sent to server');
        }
      });
    });
    
    // Listen for new incidents and refresh the dashboard
    socket.on('newIncident', () => {
      // Refetch dashboard data when a new incident is added
      
    });

    return () => socket.off('alert').off('labelAlert');
  }, []);

  return (
    <div className="app">
      <BrowserRouter>
        <ThemeProvider theme={theme}>
          <CssBaseline />
          <Routes>
             {/* Separate route for login without layout */}
             <Route path="/login" element={!isAuthenticated ? <Login /> : <Navigate to="/dashboard" replace />} />
            <Route element={<Layout />}>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />

              <Route path="/dashboard" element={isAuthenticated ? <Dashboard /> : <Navigate to="/login" replace />} />
              <Route path="/smokers" element={isAuthenticated ? <Smokers /> : <Navigate to="/login" replace />} />
              <Route path="/fighters" element={isAuthenticated ? <Fighters /> : <Navigate to="/login" replace />} />
        
            </Route>
          </Routes>
        </ThemeProvider>
      </BrowserRouter>
    </div>
  );
}

export default App;
