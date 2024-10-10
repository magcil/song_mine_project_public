import { observer } from "mobx-react-lite";
import React from "react";
import { ReactMic } from "react-mic";
import predictionsService from "services/museekService/predictionsService";
import { useStores } from "stores/rootStore";
import mic from "assets/svgs/mic.svg";
import museekMagcil from "assets/svgs/museekMagcil.svg";
import play from "assets/svgs/play.svg";
import Result from "../Result";
import "./index.css";

const AudioRecorder: React.FC = observer(() => {
  const { functionalityStore } = useStores();

  // const [blobs, setBlobs] = useState<any>([]);

  const startRecording = () => {
    functionalityStore.setRecording(true);

    functionalityStore.setResult(undefined); //clean up result after every starting

    setTimeout(() => {
      functionalityStore.setRecording(false);
    }, 1000 * 11);
  };

  const onData = () => {
    //   setBlobs(recordedBlob.blob);
    // TODO : FUTURE USE FOR REALTIME ANALYSIS
  };

  const onStop = async (recordedBlob: any) => {
    const formData = new FormData();
    formData.append("file", recordedBlob.blob);
    predictionsService
      .uploadAudio(formData)
      .then((r) => functionalityStore.setResult(r))
      .catch((e) => console.error(e));
  };
  const toggle = () => {
    if (functionalityStore.recording) {
      return;
    } else {
      startRecording();
    }
  };
  return (
    <div className="audio-player">
      <Result />
      <div className="sound-animation">
        <ReactMic
          record={functionalityStore.recording}
          className="sound-wave"
          onStop={onStop}
          onData={onData}
          strokeColor="#f9dda1"
          visualSetting="frequencyBars"
          backgroundColor="white"
          sampleRate={8000}
          mimeType="audio/wav"
        />
        <img src={museekMagcil} alt="museek Logo" />
      </div>
      <div className={`controls`} onClick={toggle}>
        <img
          src={functionalityStore.recording ? mic : play}
          id={functionalityStore.recording ? "pause" : "play"}
          alt="recording-controls"
        />
      </div>
    </div>
  );
});

export default AudioRecorder;
