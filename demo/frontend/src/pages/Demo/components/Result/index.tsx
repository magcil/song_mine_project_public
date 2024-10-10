import logo from "assets/svgs/logo.svg";
import Consts from "consts/consts";
import { observer } from "mobx-react";
import React from "react";
import { useStores } from "stores/rootStore";
import "./index.css";

const Result: React.FC = observer(() => {
  const { functionalityStore } = useStores();

  return (
    <>
      {functionalityStore.result && !functionalityStore.recording ? (
        <div className="result">
          <div className="logo">
            <img src={logo} alt="museek log" />
          </div>
          <div className="song-details">
            <span>
              Song:{" "}
              {functionalityStore.result.score < Consts.THRESHOLD
                ? "Not In Database"
                : functionalityStore.result?.winner}
            </span>
            <span>
              Score:{" "}
              {Math.round((functionalityStore.result?.score || 0) * 100) / 100}
            </span>
          </div>
        </div>
      ) : (
        <div className="empty-result"></div>
      )}
    </>
  );
});

export default Result;
