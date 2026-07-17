import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import BiteVisualizer from "../app/BiteVisualizer";
import "../app/globals.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BiteVisualizer />
  </StrictMode>,
);
