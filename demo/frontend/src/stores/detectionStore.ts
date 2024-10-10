import { makeAutoObservable } from "mobx";
import detectionService from "services/museekService/detectionService";
import {
  PaginatedResultDTO,
  PopularWinnerDTO,
  TopWinnersByDeviceDTO,
} from "services/museekService/detectionService/dtos/detectionModels";

class DetectionStore {
  paginatedResults: PaginatedResultDTO | null = null;
  mostPopularWinner: PopularWinnerDTO | null = null;
  topWinnersByDevice: TopWinnersByDeviceDTO[] = [];

  constructor() {
    makeAutoObservable(this);
  }

  getDetectedData = async (
    device?: string,
    page?: number,
    pageSize?: number
  ) => {
    const data = await detectionService.getDetected(device, page, pageSize);
    this.paginatedResults = data;
    return data;
  };

  getMostPopularWinnerData = async () => {
    const data = await detectionService.getMostPopularWinner();
    this.mostPopularWinner = data;
    return data;
  };

  getTopWinnersByDeviceData = async (topN?: number) => {
    const data = await detectionService.getTopWinnersByDevice(topN);
    this.topWinnersByDevice = data;
    return data;
  };
}

export default DetectionStore;
