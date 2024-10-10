import { makeAutoObservable } from "mobx";
import fingerprintService from "services/museekService/fingerprintService";
import songsService from "services/museekService/songsService";

class SongsStore {
  songs: any;
  totalSongs: number = 0;
  totalFingeprints: number = 0;

  constructor() {
    makeAutoObservable(this);
  }

  getAllSongs = async (skip?:number, limit?:number) => {
    // Wrap all changes with runInAction
    const result = await songsService.getAllTracks(skip, limit);
    this.songs = result.songs;
    this.totalSongs = result.totalCount;
    return result;
  };

  getTotalFingerprints = async () => {
    // Wrap all changes with runInAction
    const result = await fingerprintService.getTotalFingerprints();
    this.totalFingeprints = result.totalCount;
    return result;
  };
}

export default SongsStore;
