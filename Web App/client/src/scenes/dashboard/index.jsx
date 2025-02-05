import React, { useEffect } from 'react';
import FlexBetween from "components/FlexBetween";
import Header from "components/Header";
import { DownloadOutlined } from "@mui/icons-material";
import SmokingRoomsIcon from '@mui/icons-material/SmokingRooms';
import SportsKabaddiIcon from '@mui/icons-material/SportsKabaddi';
import GroupsIcon from '@mui/icons-material/Groups';
import { Box, Button, Typography, useTheme, useMediaQuery } from "@mui/material";
import { useGetDashboardDataQuery } from 'state/api';
import StatBox from "components/StatBox";
import BarChart from "components/BarChart";
import { DataGrid } from "@mui/x-data-grid";
import PieChart from 'components/PieChart';
import { Logout } from "@mui/icons-material"; // Import logout icon
import { useNavigate } from 'react-router-dom'; // Correctly import useNavigate
import { useDispatch } from 'react-redux';
import { clearAuthStatus } from 'state/index';
import io from 'socket.io-client';

const Dashboard = () => {
    const theme = useTheme();
    const isNonMediumScreens = useMediaQuery("(min-width: 1200px)");
    
    const { data, isLoading, refetch } = useGetDashboardDataQuery(); // Add refetch here


    const navigate = useNavigate(); // Correctly use navigate within Dashboard
    const dispatch = useDispatch();

    useEffect(() => {
      const baseURL = process.env.REACT_APP_BASE_URL;
      const socket = io(`${baseURL}`);

      // Listen for new incidents and refetch dashboard data
      socket.on('newIncident', () => {
          refetch(); // Trigger a refetch of the dashboard data
      });

      return () => socket.off('newIncident'); // Clean up the event listener
    }, [refetch]);
    
    const columns = [
      {
          field: "studentName",
          headerName: "Student Name",
          flex: 1,
      },
      {
          field: "cameraName",
          headerName: "Camera Name",
          flex: 1,
      },
      {
          field: "behaviourDetected",
          headerName: "Behavior Detected",
          flex: 1,
      },
      {
          field: "studentID",
          headerName: "Student ID",
          flex: 1,
      },
      {
          field: "date",
          headerName: "Date",
          flex: 1,
      },
      {
          field: "timestamp",
          headerName: "Timestamp",
          flex: 1,
      },
    ];
    

    // Function to handle logout
    const handleLogout = () => {
      // Clear token from local storage
      localStorage.removeItem('token');
      // Update state to reflect user is not authenticated
      dispatch(clearAuthStatus());
      // Redirect to login
      navigate('/login'); // Correctly use navigate to redirect
    };

    // Handler to trigger download
    const handleDownloadReport = () => {
        // Fetch PDF from the backend
        fetch(`${process.env.REACT_APP_BASE_URL}/reports/download`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/pdf'
            }
        })
        .then(response => {
            // Check if the response is ok
            if (response.ok) return response.blob();
            throw new Error('Network response was not ok.');
        })
        .then(blob => {
            // Create a URL for the blob and trigger download
            const url = window.URL.createObjectURL(new Blob([blob]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'report.pdf'); // Specify the file name
            document.body.appendChild(link);
            link.click();
            link.parentNode.removeChild(link);
        })
        .catch(err => console.error('Error downloading report:', err));
    };

    if (isLoading) return <Typography>Loading...</Typography>;
    if (!data) return <Typography>No data available</Typography>;

    return (
        <Box m="1.5rem 2.5rem">
          <FlexBetween>
            <Header title="DASHBOARD" subtitle="Welcome to SSS's Dashboard" />
    
            {/* Logout and Download Report Buttons */}
            <Box>
              <Button
                onClick={handleLogout}
                sx={{
                  backgroundColor: theme.palette.error.main,
                  color: theme.palette.background.alt,
                  fontSize: "14px",
                  fontWeight: "bold",
                  padding: "10px 20px",
                  marginRight: "10px",
                  "&:hover": {
                    backgroundColor: theme.palette.error[400],
                  }
                }}
              >
                Logout
              </Button>

              <Button
                onClick={handleDownloadReport}
                sx={{
                  backgroundColor: theme.palette.secondary.main,
                  color: theme.palette.background.alt,
                  fontSize: "14px",
                  fontWeight: "bold",
                  padding: "10px 20px",
                  "&:hover": {
                    backgroundColor: theme.palette.secondary[400],
                  }
                }}
              >
                <DownloadOutlined sx={{ mr: "10px" }} />
                Download Report
              </Button>
            </Box>
          </FlexBetween>
    
          <Box
                mt="20px"
                display="grid"
                gridTemplateColumns="repeat(12, 1fr)"
                gridAutoRows="160px"
                gap="20px"
                sx={{
                "& > div": { gridColumn: isNonMediumScreens ? undefined : "span 12" },
                }}
           >
            {/* ROW 1 */}
            <StatBox
                title="Smoking Incidents"
                value={data.smokingIncidentsCount}
                description={
                  <Typography sx={{ textAlign: "left" }}>
                    Total Smoking Incidents
                  </Typography>
                }
                icon={
                    <SmokingRoomsIcon sx={{ color: theme.palette.secondary[300], fontSize: "26px" }}/>
                }
            />
            <StatBox
                title="Fighting Incidents"
                value={data.fightingIncidentsCount}
                description={
                  <Typography sx={{ textAlign: "left" }}>
                    Total Fighting Incidents
                  </Typography>
                }
                icon={
                    <SportsKabaddiIcon sx={{ color: theme.palette.secondary[300], fontSize: "26px" }}/>
                }
            />

            <Box
              gridColumn="span 8"
              gridRow="span 2"
              backgroundColor={theme.palette.background.alt}
              p="1rem"
              borderRadius="0.55rem"
            >
              <BarChart title="Bar Chart - (X-axis: BehaviourDetected, Y-axis: Count)" data={data} />
            </Box>

            <StatBox
              title="Flagged Students"
              value={data.uniqueStudentCount}
              description={
                <Typography sx={{ textAlign: "left" }}>
                  Total Students Data
                </Typography>
              }
              icon={
                <GroupsIcon sx={{ color: theme.palette.secondary[300], fontSize: "26px" }} />
              }
            />


            {/* BOX 2 */}
            <Box
                gridColumn="span 8"
                gridRow="span 3"
            >
                <DataGrid
                    loading={isLoading || !data}
                    getRowId={(row) => row._id}
                    rows={(data.incidents) || []}
                    columns={columns}
                />
            </Box>
            
            <Box
              gridColumn="span 4"
              gridRow="span 3"
              backgroundColor={theme.palette.background.alt}
              p="1.5rem"
              borderRadius="0.55rem"
            >
              <Typography variant="h6" sx={{ color: theme.palette.secondary[100] }}>
                Count - Smokers and Fighters
              </Typography>
              <PieChart data={data} />
              <Typography
                p="0 0.6rem"
                fontSize="0.8rem"
                sx={{ color: theme.palette.secondary[200] }}
              >
              </Typography>
            </Box>
          </Box>
        </Box>
    );
};

export default Dashboard;
