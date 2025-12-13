import { createRoot } from "react-dom/client";
import App from "./App";

// 환경변수 확인
console.log("🔍 API_BASE:", import.meta.env.VITE_API_URL);

import "./styles/globals.css";

createRoot(document.getElementById("root")!).render(<App />);


