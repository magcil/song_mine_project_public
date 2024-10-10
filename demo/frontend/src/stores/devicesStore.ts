import { makeAutoObservable, runInAction } from "mobx";
import balenaService from "services/balenaService";
import { BalenaDevices } from "services/balenaService/dtos/balenaDevices";

class DevicesStore {
  devices: BalenaDevices[] = [];
  constructor() {
    makeAutoObservable(this);
  }

  getDevices = async () => {
    // Wrap all changes with runInAction
    runInAction(async () => {
      const devices = await balenaService.getBalenaDevices();
      this.devices = devices;
      return devices;
    });
  };
}

export default DevicesStore;
