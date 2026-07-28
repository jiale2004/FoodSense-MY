import { Routes, Route } from "react-router-dom";
import Home from "./pages/Home.jsx";
import Scan from "./pages/Scan.jsx";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/scan" element={<Scan />} />
    </Routes>
  );
}
