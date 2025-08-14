// main.tsx
import { StrictMode, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import "./global.css";
import App from "./App";
import { auth, onAuthStateChanged } from "./firebase";
import { User } from "firebase/auth";
import LoginPage from "./LoginPage";

function Root() {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [idToken, setIdToken] = useState<string | null>(null);


  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (u: User | null) => {
      setUser(u);
      setLoading(false);
      if (u) {
        u.getIdToken().then((token) => {
          setIdToken(token);
        });
      }

    });
    return () => unsubscribe();
  }, []);

  if (loading) return <div>Loading...</div>;

  return user ? <App token={idToken} /> : <LoginPage/>;
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <Root />
    </BrowserRouter>
  </StrictMode>
);
