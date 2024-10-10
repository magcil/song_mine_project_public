import React from "react";
import "./index.css";

interface LoadingProps {
  loading?: boolean;
}
const Loading: React.FC<LoadingProps> = ({ loading = false }: LoadingProps) => {
  return loading ? (
    <div className="loading-wrapper">
      <div className="loading">
        <div className="plate">
          <div className="black">
            <div className="border">
              <div className="white">
                <div className="center"></div>
              </div>
            </div>
          </div>
        </div>
        <div className="player">
          <div className="rect"></div>
          <div className="circ"></div>
        </div>
      </div>
      <div className="loading-text">
        <span className="loading-text-words">Loading</span>
      </div>
    </div>
  ) : null;
};

export default Loading;
