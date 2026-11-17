import { BrowserRouter, Routes, Route } from "react-router-dom";
//react router dom allows to create multiple pages in a app and navigate between them without reloading the page
//Routes is a container for all the Route components and ensures only one route ensures at a time
//Route defines the URL path , which component to show

import Home from "./pages/Home";
import LinearReg from "./pages/LinearReg";
import RandomForest from "./pages/RandomForest";
import Comparison from "./pages/Comparison";
import Classification from "./pages/Classification";  

//root component
//BrowserRouter wraps entire app
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/LinearRegression" element={<LinearReg />} />
        <Route path="/RandomForest" element={<RandomForest />} />
        <Route path="/Comparison" element={<Comparison />} />
        <Route path="/Classification" element={<Classification />} />  
      </Routes>
    </BrowserRouter>
  );
}

export default App;