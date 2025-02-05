const dummyStudents = [
    {
        rollNumber: "20F-0101",
        name: "Muhammad Abubakar Siddique"
    },
    {
        rollNumber: "20I-0623",
        name: "Abdullah Basit"
    },
    {
        rollNumber: "20F-0203",
        name: "Omar Ahsan Khattak"
    },
    {
        rollNumber: "20F-0296",
        name: "Shehryar Khan"
    },
    {
        rollNumber: "20F-0145",
        name: "Hassan Raza"
    },
    {
        rollNumber: "20F-0237",
        name: "Ali Ahmed"
    },
    {
        rollNumber: "20F-0110",
        name: "Ayesha Khan"
    },
    {
        rollNumber: "20F-0122",
        name: "Zainab Ali"
    },
    {
        rollNumber: "20F-0450",
        name: "Bilal Tariq"
    },
    {
        rollNumber: "20F-0333",
        name: "Faisal Imran"
    },
    {
        rollNumber: "20F-0345",
        name: "Sarah Javed"
    },
    {
        rollNumber: "20F-0222",
        name: "Hamza Malik"
    },
    {
        rollNumber: "20F-0289",
        name: "Asim Saeed"
    },
    {
        rollNumber: "20F-0351",
        name: "Mariam Yousaf"
    },
    {
        rollNumber: "20F-0399",
        name: "Usman Farooq"
    },
    {
        rollNumber: "20F-0273",
        name: "Sana Shah"
    },
    {
        rollNumber: "20F-0315",
        name: "Rizwan Akhtar"
    },
    {
        rollNumber: "20F-0488",
        name: "Nadia Aslam"
    },
    {
        rollNumber: "20F-0190",
        name: "Jawad Hussain"
    },
    {
        rollNumber: "20F-0211",
        name: "Saad Rehman"
    },
];


const dummyIncidents = [
    // Smoking Incidents (5 Entries)
    {
        time: new Date(),
        cameraName: "Camera 1",
        behaviourDetected: "Smoking",
        studentID: "20F-0101"
    },
    {
        time: new Date(),
        cameraName: "Camera 2",
        behaviourDetected: "Smoking",
        studentID: "20F-0145"
    },
    {
        time: new Date(),
        cameraName: "Camera 3",
        behaviourDetected: "Smoking",
        studentID: "20F-0110"
    },
    {
        time: new Date(),
        cameraName: "Camera 4",
        behaviourDetected: "Smoking",
        studentID: "20F-0345"
    },
    {
        time: new Date(),
        cameraName: "Camera 1",
        behaviourDetected: "Smoking",
        studentID: "20F-0450"
    },

    // Fighting Incidents (15 Entries)
    {
        time: new Date(),
        cameraName: "Camera 1",
        behaviourDetected: "Fighting",
        studentID: "20I-0623"
    },
    {
        time: new Date(),
        cameraName: "Camera 2",
        behaviourDetected: "Fighting",
        studentID: "20F-0203"
    },
    {
        time: new Date(),
        cameraName: "Camera 3",
        behaviourDetected: "Fighting",
        studentID: "20F-0296"
    },
    {
        time: new Date(),
        cameraName: "Camera 4",
        behaviourDetected: "Fighting",
        studentID: "20F-0237"
    },
    {
        time: new Date(),
        cameraName: "Camera 2",
        behaviourDetected: "Fighting",
        studentID: "20F-0122"
    },
    {
        time: new Date(),
        cameraName: "Camera 3",
        behaviourDetected: "Fighting",
        studentID: "20F-0333"
    },
    {
        time: new Date(),
        cameraName: "Camera 4",
        behaviourDetected: "Fighting",
        studentID: "20F-0222"
    },
    {
        time: new Date(),
        cameraName: "Camera 1",
        behaviourDetected: "Fighting",
        studentID: "20F-0289"
    },
    {
        time: new Date(),
        cameraName: "Camera 2",
        behaviourDetected: "Fighting",
        studentID: "20F-0351"
    },
    {
        time: new Date(),
        cameraName: "Camera 3",
        behaviourDetected: "Fighting",
        studentID: "20F-0399"
    },
    {
        time: new Date(),
        cameraName: "Camera 4",
        behaviourDetected: "Fighting",
        studentID: "20F-0273"
    },
    {
        time: new Date(),
        cameraName: "Camera 1",
        behaviourDetected: "Fighting",
        studentID: "20F-0315"
    },
    {
        time: new Date(),
        cameraName: "Camera 2",
        behaviourDetected: "Fighting",
        studentID: "20F-0488"
    },
    {
        time: new Date(),
        cameraName: "Camera 3",
        behaviourDetected: "Fighting",
        studentID: "20F-0190"
    },
    {
        time: new Date(),
        cameraName: "Camera 4",
        behaviourDetected: "Fighting",
        studentID: "20F-0211"
    },
];



const logindata = [
    {
        Username: "alpha_ab",
        password: "Ryzen_204b"  // Change 'Password' to 'password'
    },
];

export {dummyStudents, dummyIncidents, logindata};