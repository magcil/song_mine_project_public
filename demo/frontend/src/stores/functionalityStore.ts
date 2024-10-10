import { makeAutoObservable } from "mobx";
import { Result } from "services/museekService/predictionsService/dto/result";

class FunctionalityStore {
  loading: boolean = false;
  recording: boolean = false;
  result: Result | undefined = undefined;
  constructor() {
    makeAutoObservable(this);
  }

  public triggerActivity = (value: boolean) => {
    this.loading = value;
  };
  toggleActivity = () => {
    this.loading = !this.loading;
  };

  setRecording = (value: boolean) => {
    this.recording = value;
  };
  toggleRecording = () => {
    this.recording = !this.recording;
  };
  setResult = (value: Result | undefined) => {
    this.result = value;
  };
}

export default FunctionalityStore;
