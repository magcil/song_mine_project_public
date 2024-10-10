import React from "react";
import { useHistory } from "react-router-dom";
import AudioRecorder from "./components/AudioRecorder";
import "./index.css";

const Demo: React.FC = () => {
  const history = useHistory();
  return (
    <div className="content">
      <AudioRecorder />
      <div className="dashboard">
        <a
          onClick={() => {
            history.push("/sign-in");
          }}
        >
          Go to Dashboard
        </a>
      </div>
    </div>
  );
};

export default Demo;
