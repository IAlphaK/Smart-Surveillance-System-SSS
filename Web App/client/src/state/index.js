import { createSlice } from "@reduxjs/toolkit";

// Initial state including mode, authentication status, and token
const initialState = {
    mode: "dark",
    isAuthenticated: false,  // Store authentication status
    token: null,             // Store token
};

export const globalSlice = createSlice({
    name: "global",
    initialState,
    reducers: {
        // Toggle between light and dark mode
        setMode: (state) => {
            state.mode = state.mode === 'light' ? "dark" : 'light';
        },
        // Set authentication status and token
        setAuthStatus: (state, action) => {
            state.isAuthenticated = action.payload.isAuthenticated;
            state.token = action.payload.token;
        },
        // Clear authentication status and token on logout
        clearAuthStatus: (state) => {
            state.isAuthenticated = false;
            state.token = null;
        }
    }
})

// Export actions for use in components
export const { setMode, setAuthStatus, clearAuthStatus } = globalSlice.actions;

export default globalSlice.reducer;
