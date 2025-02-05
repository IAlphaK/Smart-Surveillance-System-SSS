import React from 'react';
import {Box, useTheme} from "@mui/material";
import { useGetFightersQuery, useGetSmokersQuery } from 'state/api';
import Header from "components/Header";
import {DataGrid} from "@mui/x-data-grid";


const Smokers = () => {
  const theme = useTheme();
  const {data, isLoading} = useGetFightersQuery();
  console.log('data', data);

  const columns = [
    {
        field: "studentName",
        headerName: "Student Name",
        flex: 1,
    },
    {
        field: "cameraName",
        headerName: "Camera Name",
        flex:1,
    },
    {
        field: "behaviourDetected",
        headerName: "Behavior Detected",
        flex:1
    },
    {
        field: "studentID",
        headerName: "Student ID",
        flex:1
    },
    {
        field: "date",
        headerName: "Date",
        flex:1
    },
    {
        field: "timestamp",
        headerName: "Timestamp",
        flex:1
    },
  ];
  
  
  return (
    <Box m="1.5rem 2.5rem">
      <Header title="Fighters" subtitle="List of Fighters"/>
      <Box
      >
          {/*Data Grid */}
          <DataGrid
            loading={isLoading || !data}
            getRowId={(row) => row._id}
            rows={data || []}
            columns={columns}
          />
      </Box>
    </Box>
  )
}

export default Smokers;