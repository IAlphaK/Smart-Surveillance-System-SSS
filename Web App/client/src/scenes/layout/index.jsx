import React, { useState } from 'react';
import { Box, useMediaQuery } from "@mui/material";
import { Outlet } from "react-router-dom";
import Navbar from "components/Navbar";
import SideBar from "components/Sidebar";
import { useGetUserQuery } from 'state/api';
import { useSelector } from 'react-redux';

const Layout = () => {
  const isNonMobile = useMediaQuery("(min-width: 600px)");
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const studentId = useSelector((state) => state.global.studentId);
  const{data} = useGetUserQuery(studentId);
  console.log("data:", data);
  

  return (
    <Box display={isNonMobile ? "flex": "block"} width="100%" height="100%">
      <SideBar
        isNonMobile={isNonMobile}
        drawerWidth="250px"
        isSidebarOpen={isSidebarOpen}
        setIsSidebarOpen={setIsSidebarOpen} // Corrected prop name
      />
      <Box flexGrow={1}>
        <Navbar setIsSidebarOpen={setIsSidebarOpen} isSidebarOpen={isSidebarOpen} />
        <Outlet/>
      </Box>
    </Box>
  );
};

export default Layout;