// firebase.js

import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
// import { getAuth } from "firebase/auth"; // si querés usar autenticación
// import { getStorage } from "firebase/storage"; // si querés subir archivos

const firebaseConfig = {
  apiKey: "AIzaSyD6u0AsNJ-Zq7o7UuaovY4kxGavoxDBBSM",
  authDomain: "diagnostico-medicodb.firebaseapp.com",
  projectId: "diagnostico-medicodb",
  storageBucket: "diagnostico-medicodb.firebasestorage.app",
  messagingSenderId: "534466090567",
  appId: "1:534466090567:web:525e60f17c6a1f2ecbc362",
  measurementId: "G-SHSGSWTKE3"
};

const app = initializeApp(firebaseConfig);
const db = getFirestore(app); // conexión a Firestore

export { db };
