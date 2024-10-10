// RootStore.js
import React from "react";
import DevicesStore from "./devicesStore";
import FunctionalityStore from "./functionalityStore";
import SongsStore from "./songsStore";
import UserStore from "./userStore";
import DetectionStore from "./detectionStore";

class RootStore {
  userStore: UserStore;
  deviceStore: DevicesStore;
  functionalityStore: FunctionalityStore;
  songsStore: SongsStore;
  detectionStore: DetectionStore;
  constructor() {
    this.userStore = new UserStore();
    this.deviceStore = new DevicesStore();
    this.functionalityStore = new FunctionalityStore();
    this.songsStore = new SongsStore();
    this.detectionStore = new DetectionStore();
  }
}

export default RootStore;

const StoresContext = React.createContext(new RootStore());

// this will be the function available for the app to connect to the stores
export const useStores = () => React.useContext(StoresContext);
