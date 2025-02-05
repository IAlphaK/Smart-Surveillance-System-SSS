import {createApi, fetchBaseQuery} from "@reduxjs/toolkit/query/react";

export const api = createApi({
    baseQuery: fetchBaseQuery({baseUrl: process.env.REACT_APP_BASE_URL}),
    reducerPath: "adminApi",
    tagTypes: ["Student"],
    endpoints: (build) => ({
        getUser: build.query({
            query: (id) => `/summary/student/${id}`,
        }),
        getSmokers: build.query({
            query: () => "/summary/smokers",
        }), 
        getFighters: build.query({
            query: () => "/summary/fighters",
        }),
        getDashboardData: build.query({
            query: () => "/general/dashboard",
            providesTags: ["Student", "Incident"]
        }),
        getReport: build.query({
            query: () => "/reports/download",
        }),        
    }),
});

export const{
    useGetUserQuery,
    useGetSmokersQuery,
    useGetFightersQuery,
    useGetDashboardDataQuery 
    /*useGetIncidentsQuery*/
} = api;