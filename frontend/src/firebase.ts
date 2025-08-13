// src/firebase.js
import { initializeApp } from "firebase/app";
import { getAuth, RecaptchaVerifier, signInWithPhoneNumber, onAuthStateChanged, signOut } from "firebase/auth";

const firebaseConfig = {
    apiKey: "AIzaSyB2r13V8v3bm6jrSX8apZnX1HZSOwqFQ0I",
    authDomain: "townplanmap.com",
    projectId: "townplanmap",
    storageBucket: "townplanmap.appspot.com",
    messagingSenderId: "585432733039",
    appId: "1:585432733039:web:137d2b5633d4662748849a"
  };

const app = initializeApp(firebaseConfig);
export const auth = getAuth(app);
auth.useDeviceLanguage();

export { RecaptchaVerifier, signInWithPhoneNumber, onAuthStateChanged, signOut };
