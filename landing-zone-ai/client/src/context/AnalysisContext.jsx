import React, { createContext, useState, useContext } from 'react';

const AnalysisContext = createContext();

export const AnalysisProvider = ({ children }) => {
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [history, setHistory] = useState([]);

  return (
    <AnalysisContext.Provider value={{ currentAnalysis, setCurrentAnalysis, history, setHistory }}>
      {children}
    </AnalysisContext.Provider>
  );
};

export const useAnalysis = () => useContext(AnalysisContext);
